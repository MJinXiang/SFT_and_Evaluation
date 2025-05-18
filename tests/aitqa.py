
import json
import os
import logging
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from utils.parser import extract_program
import pandas as pd
import numpy as np
from prompt import COT_PROMPT_AITQA_TEMPLATE, COT_PROMPT_TEMPLATE

# pd.set_option('display.expand_frame_repr', False)
# pd.set_option('display.max_rows', 8)
# pd.set_option('display.max_columns', 10)

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 500000)
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Setup logging
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('tatqa_processor')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class BaseGenerator:
    """统一的生成器接口"""
    def generate(self, prompts, max_new_tokens=2048, temperature=0.0, top_p=1.0, verbose=False, batch_size=32):
        raise NotImplementedError("Subclasses must implement this method.")


class VLLMGenerator(BaseGenerator):  
    def __init__(self, model_path, max_model_len=8192, tensor_parallel_size=1):  
        self.EOS = ["<|im_end|>", "</s>"]  
        self.model = LLM(  
            model=model_path,  
            max_model_len=max_model_len,  
            trust_remote_code=True,  
            gpu_memory_utilization=0.90, 
            distributed_executor_backend='ray',  
            tensor_parallel_size=tensor_parallel_size
        )  
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def generate(self, prompts, max_new_tokens=2048, temperature=0.0, top_p=1.0, verbose=False, batch_size=32):
        try:  
            chat_prompts = []  
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                chat_prompts.append(self.tokenizer.apply_chat_template(  
                    messages, tokenize=False, add_generation_prompt=True))  
            
            vllm_outputs = self.model.generate(  
                prompts=chat_prompts,  
                sampling_params=SamplingParams(  
                    max_tokens=max_new_tokens,  
                    temperature=temperature,  
                    top_p=top_p,  
                    stop=self.EOS + ["<output>"],  
                ),  
                use_tqdm=True,  
            )  
            
            return [x.outputs[0].text for x in vllm_outputs]  
        except Exception as e:  
            print(f"Error in vLLM generation: {str(e)}")  
            raise


import concurrent.futures
from threading import Lock

class APIGenerator(BaseGenerator):
    """基于API的生成器"""
    def __init__(self, model_info):
        from utils.llm import initialize_client, call_api_with_retry
        self.client_info = initialize_client(model_info)
        self.call_api = call_api_with_retry

        # 添加tokenizer以便计算token数量
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("/netcache/huggingface/Qwen2.5-3B-Instruct")
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            self.tokenizer = None

    def _process_single_prompt(self, prompt_with_index):
        """处理单个请求，用于线程池调用"""
        idx, prompt = prompt_with_index
        messages = [{"role": "user", "content": prompt}]
        
        try:
            success, response, *thinking = self.call_api(
                client_info=self.client_info,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                max_retries=10
            )
            
            if not success:
                return idx, None, f"API call failed: {response}"
            
            if self.client_info["model_type"] == "openai":
                result = response.choices[0].message.content
            else:
                result = response
                
            return idx, result, None
        except Exception as e:
            return idx, None, str(e)
    
    def generate(self, prompts, max_new_tokens=2048, temperature=0.0, top_p=1.0, verbose=False, batch_size=32):
        """使用线程池并行处理API请求"""
        self.max_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        results = [None] * len(prompts)
        errors = []
        
        # 创建提示与索引的映射
        prompts_with_indices = list(enumerate(prompts))
        
        with tqdm(total=len(prompts), desc="API Generation") as pbar:
            # 使用线程池并行处理
            with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_prompt = {
                    executor.submit(self._process_single_prompt, prompt_tuple): prompt_tuple 
                    for prompt_tuple in prompts_with_indices
                }
                
                for future in concurrent.futures.as_completed(future_to_prompt):
                    idx, result, error = future.result()
                    if error:
                        errors.append((idx, error))
                        results[idx] = f"ERROR: {error}"
                    else:
                        results[idx] = result
                    pbar.update(1)
        
        # 处理所有错误
        if errors:
            print(f"Encountered {len(errors)} errors during API calls")
            for idx, error in errors:
                print(f"Error for prompt {idx}: {error}")
                
        return results


def create_generator(model_config):
    
    """根据配置创建适当的生成器"""
    use_api = model_config.get("use_api", False)
    
    if use_api:
        return APIGenerator(model_config)
    else:
        return VLLMGenerator(
            model_path=model_config["model_path"],
            max_model_len=model_config.get("max_model_len", 8192),
            tensor_parallel_size=model_config.get("tensor_parallel_size", 1)
        )


def prepare_item(item, few_shot_prompts):
    """为TAT-QA任务准备项目"""
    df_prompt = item.get("df_prompt", "")
    df_code = item.get("df_code", "")
    question = item.get("question", "")
    
    # 处理段落信息
    paragraphs = []
    if "prompt_info" in item and "paragraphs" in item["prompt_info"]:
        paragraph_list = item["prompt_info"]["paragraphs"]
        # 按顺序排序段落
        sorted_paragraphs = sorted(paragraph_list, key=lambda p: p.get("order", 0))
        paragraphs = [p.get("text", "") for p in sorted_paragraphs]
    
    # 合并段落文本
    paragraphs_text = "\n\n".join(paragraphs)

    local_namespace = {}
    
    # 如果有df_code，执行它来获取DataFrame
    if df_code:
        try:
            exec(df_code, globals(), local_namespace)
            # 调用函数获取DataFrame
            df = local_namespace['get_dataframe']()
        except Exception as e:
            print(f"Error executing df_code: {e}")
            df = None

    # 将DataFrame转为字符串表示
    if df is not None:
        df_string = df.to_string(index=True)
    else:
        df_string = "DataFrame could not be created"
    
    # 使用few-shot模板格式化提示
   
    prompt = few_shot_prompts.format(
            df_prompt=df_prompt,
            df=df_string,
            paragraphs=paragraphs_text,
            question=question
        )
    
    # 添加到项目中
    item["prompt"] = prompt
    item["df"] = df
    item["paragraphs"] = paragraphs
    
    return item

def prepare_wo_ex_item(item, few_shot_prompts):
    """为TAT-QA任务准备项目"""
    df_prompt = item.get("df_prompt", "")
    df_code = item.get("df_code", "")
    question = item.get("question", "")
    
    # 处理段落信息
    paragraphs = []
    if "prompt_info" in item and "paragraphs" in item["prompt_info"]:
        paragraph_list = item["prompt_info"]["paragraphs"]
        # 按顺序排序段落
        sorted_paragraphs = sorted(paragraph_list, key=lambda p: p.get("order", 0))
        paragraphs = [p.get("text", "") for p in sorted_paragraphs]
    
    # 合并段落文本
    paragraphs_text = "\n\n".join(paragraphs)

    local_namespace = {}
    
    # 如果有df_code，执行它来获取DataFrame
    if df_code:
        try:
            exec(df_code, globals(), local_namespace)
            # 调用函数获取DataFrame
            df = local_namespace['get_dataframe']()
        except Exception as e:
            print(f"Error executing df_code: {e}")
            df = None

    # 将DataFrame转为字符串表示
    if df is not None:
        df_string = df.to_string(index=True)
    else:
        df_string = "DataFrame could not be created"
    
    # 使用few-shot模板格式化提示
    df_code_block = f"```python\n{df_code}\n```" 
   
    prompt = few_shot_prompts.format(
            # table=df_code_block,
            table=df_string,
            text=paragraphs_text,
            question=question
        )
    
    # 添加到项目中
    item["prompt"] = prompt
    item["df"] = df
    item["paragraphs"] = paragraphs
    
    return item


# def prepare_markdown_item(item, markdown_prompts=None):
#     """为 TABLEBENCH 任务准备 markdown 项目，支持 column_header + row_header 表格结构"""
#     table_data = None
#     column_header = []
#     row_header = []
#     question = None

#     # 解析表格数据
#     if "table" in item:
#         table = item["table"]
#         if isinstance(table, dict):
#             column_header = table.get("column_header", [])
#             row_header = table.get("row_header", [])
#             table_data = table.get("data", [])
#         elif isinstance(table, list):
#             table_data = table
#         elif isinstance(table, dict) and "table" in table:
#             table_data = table["table"]

#     # 解析问题文本
#     if "question" in item:
#         question = item["question"]
#     elif "question_item" in item and "question" in item["question_item"]:
#         question = item["question_item"]["question"]

#     # 处理列标题（提取每列的标签）
#     if column_header and isinstance(column_header[0], list):
#     # 合并每个子列表中的元素，如 ["Increase (decrease) from 2018 (a):", "Domestic"] 
#     # 变为 "Increase (decrease) from 2018 (a): Domestic"
#         columns = [f"{col[0]} {col[1]}" if len(col) > 1 else col[0] for col in column_header]
#     else:
#         columns = column_header
#     # if column_header and isinstance(column_header[0], list):
#     #     columns = [col[0] for col in column_header]
#     # else:
#     #     columns = column_header

#     # 如果 row_header 存在，插入 row_header 列标题
#     if row_header and isinstance(row_header[0], list):
#         columns = ["Row Header"] + columns
#         # 合并 row_header 和 data 行
#         full_data = []
#         for rh, row in zip(row_header, table_data):
#             row_label = " / ".join(rh)
#             full_data.append([row_label] + row)
#     else:
#         full_data = table_data

#     # 构建 Markdown 表格字符串
#     table_str = ""
#     if columns:
#         table_str += " | ".join(columns) + "\n"
#         table_str += " | ".join(["---"] * len(columns)) + "\n"
#     if full_data:
#         for row in full_data:
#             table_str += " | ".join(str(cell) for cell in row) + "\n"

#     # 生成提示
#     prompt = markdown_prompts.format(
#         table=table_str,
#         question=question
#     )

#     # 创建新的结果对象
#     result = item.copy()
#     result["prompt"] = prompt
#     result["df"] = None  # 不使用DataFrame
#     return result

def prepare_markdown_item(item, markdown_prompts=None):
    """为 TABLEBENCH 任务准备 markdown 项目，支持 column_header + row_header 表格结构"""
    table_data = None
    column_header = []
    row_header = []
    question = None

    # 解析表格数据
    if "table" in item:
        table = item["table"]
        if isinstance(table, dict):
            column_header = table.get("column_header", [])
            row_header = table.get("row_header", [])
            table_data = table.get("data", [])
        elif isinstance(table, list):
            table_data = table
        elif isinstance(table, dict) and "table" in table:
            table_data = table["table"]

    # 解析问题文本
    if "question" in item:
        question = item["question"]
    elif "question_item" in item and "question" in item["question_item"]:
        question = item["question_item"]["question"]

    # 格式化原始表头信息，用于在提示中展示
    column_headers_str = json.dumps(column_header, indent=2) if column_header else "[]"
    row_headers_str = json.dumps(row_header, indent=2) if row_header else "[]"

    # 处理列标题（提取每列的标签）
    if column_header and isinstance(column_header[0], list):
        # 合并每个子列表中的元素
        columns = [f"{col[0]} {col[1]}" if len(col) > 1 else col[0] for col in column_header]
    else:
        columns = column_header

    # 如果 row_header 存在，插入 row_header 列标题
    if row_header and isinstance(row_header[0], list):
        columns = ["Row Header"] + columns
        # 合并 row_header 和 data 行
        full_data = []
        for rh, row in zip(row_header, table_data):
            row_label = " / ".join(rh)
            full_data.append([row_label] + row)
    else:
        full_data = table_data

    # 构建 Markdown 表格字符串
    table_str = ""
    if columns:
        table_str += " | ".join(columns) + "\n"
        table_str += " | ".join(["---"] * len(columns)) + "\n"
    if full_data:
        for row in full_data:
            table_str += " | ".join(str(cell) for cell in row) + "\n"

    # 生成提示
    prompt = markdown_prompts.format(
        table=table_str,
        column_headers=column_headers_str,
        row_headers=row_headers_str,
        question=question
    )

    # 创建新的结果对象
    result = item.copy()
    result["prompt"] = prompt
    result["df"] = None  # 不使用DataFrame
    return result

def execute_with_dataframe(code, df):
    """执行给定的Python代码，提供DataFrame作为输入"""
    # 创建带有pandas和DataFrame的命名空间
    namespace = {
        'pd': pd,  # Make pandas available in the code
        'np': np,
        'df': df   # Provide the dataframe
    }
    
    try:
        exec(code, namespace)
        
        function_name = next((name for name, obj in namespace.items() 
                           if callable(obj) and name != 'print' and not name.startswith('__')), None)
        
        if function_name:
            result = namespace[function_name](df)
            return result, None
        else:
            return None, "No function found in code"
    except Exception as e:
        return None, f"Error executing code: {str(e)}"


def extract_answer(text):
    """从文本中提取答案"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def process_tablebench_data_batch(input_file, output_file, model_config, log_file, max_tokens=2048,
                         temperature=0.0, tensor_parallel_size=1, start_from=0, batch_size=32):
    logger = setup_logger(log_file)
    start_time = time.time()
    logger.info(f"Started processing TAT-QA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input: {input_file}, Output: {output_file}, Model: {model_config['model_path']}")

    # 记录使用的提示模式
    prompt_mode = model_config.get("prompt_mode", "with_execution")
    logger.info(f"Using prompt mode: {prompt_mode}")
    
    try:
        generator = create_generator(model_config)
        logger.info(f"Generator initialized successfully: {'API' if model_config.get('use_api', False) else 'VLLM'}")
    except Exception as e:
        logger.error(f"Generator initialization failed: {e}")
        return
    
    # 读取数据项
    try:
        data_items = []
        # 检查文件扩展名
        if input_file.lower().endswith('.jsonl'):
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # 跳过空行
                        try:
                            item = json.loads(line)
                            data_items.append(item)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse line in JSONL: {e}")
            logger.info(f"Loaded {len(data_items)} data items from JSONL file")
        else:
            # 处理常规JSON文件
            with open(input_file, 'r', encoding='utf-8') as f:
                data_items = json.load(f)
            logger.info(f"Loaded {len(data_items)} data items from JSON file")
        
        if not data_items:
            logger.error("No valid data items found in the input file")
            return
            
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
    # 检查中间结果
    results = []
    processed_ids = set()
    if os.path.exists(f"{output_file}.temp"):
        try:
            with open(f"{output_file}.temp", 'r', encoding='utf-8') as f:
                temp_data = json.load(f)
                if isinstance(temp_data, dict) and "results" in temp_data:
                    results = temp_data["results"]
                else:
                    results = temp_data
                
            processed_ids = {result.get("uid", "") for result in results}
            logger.info(f"Loaded {len(results)} intermediate results")
        except Exception as e:
            logger.error(f"Failed to load intermediate results: {e}")
            results = []
            processed_ids = set()
    
    remaining_items = [item for idx, item in enumerate(data_items) 
                      if idx >= start_from and item.get("uid", f"item_{idx}") not in processed_ids]
    
    logger.info(f"Remaining items to process: {len(remaining_items)}/{len(data_items)}")
    
    success_count = len(results)
    error_count = 0

    # 一次处理所有数据
    if prompt_mode == "with_execution":
        logger.info("Preparing items with code execution support")
        all_items = [(i, prepare_item(item, FEW_SHOT_PROMPTS)) 
                    for i, item in enumerate(remaining_items)]
    elif prompt_mode == "markdown":
        logger.info("Preparing items with markdown table format")
        all_items = [(i, prepare_markdown_item(item, COT_PROMPT_AITQA_TEMPLATE)) 
                    for i, item in enumerate(remaining_items)]
    else:  # wo_execution
        logger.info("Preparing items without code execution")
        all_items = [(i, prepare_wo_ex_item(item, COT_PROMPT_TEMPLATE)) 
                    for i, item in enumerate(remaining_items)]

    
    all_prompts = [(i, item["prompt"]) for i, item in all_items]
    end_prompts = []
    remain_prompts = []
    # 存储不带提示的模型响应
    model_responses = {i: "" for i, _ in all_prompts}
    
    max_func_call = 5  # 函数调用的最大次数
    max_tokens_allowed = 16384  # 安全缓冲区
    
    # 运行执行循环
    for epoch in range(max_func_call):
        logger.info(f"Execution epoch {epoch+1}/{max_func_call}")
        current_prompts = all_prompts if epoch == 0 else remain_prompts
        if not current_prompts:
            break
        
        prompts = [item[1] for item in current_prompts]
        prompt_ids = [item[0] for item in current_prompts]
        
        # 在发送到模型之前检查令牌长度
        prompts_to_process = []
        skipped_indices = []
        
        for idx, (prompt_idx, prompt) in enumerate(zip(prompt_ids, prompts)):
            try:
                token_count = len(generator.tokenizer.encode(prompt))
                if token_count > max_tokens_allowed:
                    logger.warning(f"Prompt {prompt_idx} is too long: {token_count} tokens, skipping")
                    error_msg = f"\n\nError: Prompt exceeded maximum allowed token length ({token_count} tokens)."
                    model_responses[prompt_idx] += error_msg
                    skipped_indices.append((idx, prompt_idx, prompt + error_msg))
                else:
                    prompts_to_process.append((idx, prompt))
            except Exception as e:
                logger.warning(f"Error calculating token length for prompt {prompt_idx}: {e}")
                prompts_to_process.append((idx, prompt))  # 在出错时仍然包含
        
        # 将跳过的提示添加到end_prompts
        for _, prompt_idx, full_prompt in skipped_indices:
            end_prompts.append((prompt_idx, full_prompt))
        
        if not prompts_to_process:
            logger.warning("No prompts to process in this batch after length filtering")
            continue
        
        # 为模型准备提示
        prompts_for_model = [p for _, p in prompts_to_process]
        prompt_indices = [i for i, _ in prompts_to_process]
        
        try:
            # 使用模型生成响应
            logger.info(f"Generating responses for {len(prompts_for_model)} prompts...")
            responses = generator.generate(
                prompts=prompts_for_model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                batch_size=batch_size
            )
            
            # 为下一次迭代重置
            remain_prompts = []
            remain_codes = []
            remain_items = []
            
            # 处理每个响应
            for orig_idx, response in zip(prompt_indices, responses):
                i = prompt_ids[orig_idx]
                query = prompts[orig_idx]
                output = response.rstrip()
                model_responses[i] += output
                full_query = query + output
                
                # 检查此输出是否太长，无法进一步处理
                combined_token_count = len(generator.tokenizer.encode(full_query))
                if combined_token_count > max_tokens_allowed - 1000:  # 为一个响应留出空间
                    logger.warning(f"Response for prompt {i} would be too long for further processing ({combined_token_count} tokens)")
                    model_responses[i] += f"\n\nWarning: Response too long for further processing. Finalizing result."
                    end_prompts.append((i, full_query))
                    continue
                
                if not (output.endswith("</think>") or output.endswith("</answer>") or output.endswith("```")):
                    end_prompts.append((i, full_query))
                elif "<answer>" not in output:
                    program = extract_program(full_query)
                    remain_prompts.append((i, full_query))
                    remain_codes.append(program)
                    item = next((item for idx, item in all_items if idx == i), None)
                    remain_items.append(item)
                else:
                    end_prompts.append((i, full_query))

            # 执行需要额外处理的提示的代码
            if remain_codes:
                logger.info(f"Executing {len(remain_codes)} code snippets")
                
                remain_results = []
                for k in range(len(remain_codes)):
                    code = remain_codes[k]
                    item = remain_items[k]
                    
                    try:
                        df = item["df"]
                        res, report = execute_with_dataframe(code, df)
                        remain_results.append((res, report))
                    except Exception as e:
                        remain_results.append((None, f"Error preparing DataFrame: {str(e)}"))
                
                # 使用执行结果更新提示
                for k in range(len(remain_prompts)):
                    i, query = remain_prompts[k]
                    res, report = remain_results[k]
                    
                    exec_result = f"\n<output>\n{res}\n</output>\n" if res is not None else f"\n<output>\nExecution error:\n{report}\n</output>\n"
                    model_responses[i] += exec_result
                    query += exec_result
                    
                    # 添加执行结果后再次检查令牌长度
                    token_count = len(generator.tokenizer.encode(query))
                    if token_count > max_tokens_allowed - 1000:
                        logger.warning(f"Prompt {i} became too long after execution: {token_count} tokens")
                        message = "\nReached maximum token length. Finalizing response."
                        query += message
                        model_responses[i] += message
                        end_prompts.append((i, query))
                        continue

                    if model_responses[i].count("```python") >= 3:
                        message = "\nYou have exceeded the maximum number of Python code usage attempts. You are not allowed to use Python Tool. Do NOT use Python Tool again!!! Please think step-by-step and give me the final answer directly. Please include your final answer inside <answer>...</answer> tags.\n\n"
                        query += message
                        model_responses[i] += message
                        remain_prompts[k] = (i, query)
                    elif epoch == max_func_call - 1:
                        message = "\nReach max function call limit."
                        query += message
                        model_responses[i] += message
                        end_prompts.append((i, query))
                    else:
                        remain_prompts[k] = (i, query)
                        
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {e}")
            # 仅将受影响的提示添加到end_prompts，而不是所有当前提示
            for i, query in current_prompts:
                if i not in [ep[0] for ep in end_prompts]:  # 仅当尚未在end_prompts中时
                    error_msg = f"\n\nError occurred during processing: {str(e)}\n"
                    model_responses[i] += error_msg
                    query += error_msg
                    end_prompts.append((i, query))
            # 不要中断 - 继续处理剩余的提示
    
    # 处理最终结果
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    
    # 记录处理的项目
    for i, (_, item) in enumerate(all_items):
        item_start_time = time.time()
        result_idx = next((idx for idx, (j, _) in enumerate(end_prompts) if j == i), -1)
        
        if result_idx != -1:
            full_response = model_responses[i]
            
            # 从<answer>和</answer>标签之间提取答案
            extracted_answer = extract_answer(full_response)
            
            item_uid = item.get("id", f"item-{i}")
            question = item.get("question", "")
            gold_answer = item.get("answers", [])  
            type = item.get("type", "")
           
            
            item_time = time.time() - item_start_time

            # 计算prompt的token长度
            prompt_token_length = 0
            try:
                prompt_token_length = len(generator.tokenizer.encode(item["prompt"]))
            except Exception as e:
                logger.warning(f"Could not calculate prompt token length: {e}")
            
            # 计算full_response的token长度
            response_token_length = 0
            try:
                response_token_length = len(generator.tokenizer.encode(full_response))
            except Exception as e:
                logger.warning(f"Could not calculate response token length: {e}")
            
            # 构建结果对象 - 确保字段名与评估脚本匹配
            result = {
                "id": item_uid,  # 使用"id"而不是"uid"作为主键，以匹配评估脚本的预期
                "question": question,
                "gold_answer": gold_answer,  # 将原始答案命名为gold_answer
                "model_answer": extracted_answer,  # 提取的答案作为model_answer
                "full_response": full_response,  # 完整响应
                "type": type,
                "processing_time": item_time,
                "prompt": item.get("prompt", ""),  # 保存原始提示以便调试
                "prompt_token_length": prompt_token_length,
                "response_token_length": response_token_length,
                "total_token_length": prompt_token_length + response_token_length
            }
            
            results.append(result)
            success_count += 1
            
            # 每处理10个项目保存一次中间结果
            if (i + 1) % 100 == 0:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results - {len(results)}/{len(data_items)} items processed")
        else:
            logger.error(f"Failed to find result for item {i}")
            error_count += 1

    # 保存最终结果
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
        
        # 移除临时文件
        if os.path.exists(f"{output_file}.temp"):
            os.remove(f"{output_file}.temp")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")
        # 尝试保存到临时文件
        try:
            backup_file = f"{output_file}.backup"
            with open(backup_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Results saved to backup file: {backup_file}")
        except Exception as e2:
            logger.error(f"Failed to save backup file: {e2}")

        # 日志摘要
        total_time = time.time() - start_time
        logger.info("=" * 60)
        logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
        logger.info(f"Successfully processed: {success_count}/{len(data_items)}, Failures: {error_count}")
        if data_items:
            logger.info(f"Success rate: {success_count/len(data_items)*100:.2f}%")
        average_time = total_time / len(remaining_items) if remaining_items else 0
        logger.info(f"Average processing time: {average_time:.2f} seconds per item")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process TAT-QA dataset with batch inference')
    
    parser.add_argument('--input_file', type=str, help='Path to input file (defaults to test.jsonl in data/tatqa)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save results')
    parser.add_argument('--model_path', type=str, required=True, help='Model path or identifier')
    parser.add_argument('--log_file', type=str, required=True, help='Path to log file')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for model generation')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum tokens for model output')
    parser.add_argument('--tensor_parallel_size', type=int, default=2, help='Tensor parallelism size')
    parser.add_argument('--start_from', type=int, default=0, help='Start processing from this index')
    parser.add_argument('--base_path', type=str, help='Base path for the project')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing (affects API concurrent requests)')
    parser.add_argument('--use_api', action='store_true', help='Use API instead of VLLM for generation')
    parser.add_argument('--api_type', type=str, default="openai", 
                        choices=["openai", "gemini", "claude", "deepseek-r1", "deepseek-r1-inner"],
                        help='API type when using --use_api')
    parser.add_argument('--prompt_mode', type=str, default="markdown", 
                    choices=["with_execution", "wo_execution", "markdown"],
                    help='Prompt preparation mode: with_execution (with code execution), wo_execution (without execution), or markdown (uses original table format)')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # 处理base_path
    base_path = None
    if args.base_path and os.path.exists(args.base_path):
        base_path = args.base_path
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(current_dir) == "tests":
            base_path = os.path.dirname(current_dir)
    
    if not base_path:
        print("Error: Unable to find project root directory")
        exit(1)
    
    print(f"Using root path: {base_path}")
    
    # 设置文件路径
    input_file = args.input_file if args.input_file else os.path.join(base_path, "data/aitqa/test.jsonl")

    # 加载few-shot提示
    global FEW_SHOT_PROMPTS
    try:
        with open(os.path.join(base_path, "few_shots/table_r1_think_tablebench.md"), 'r', encoding='utf-8') as f:
            FEW_SHOT_PROMPTS = f.read()
    except Exception as e:
        print(f"Warning: Could not load few-shot prompts: {e}")
        FEW_SHOT_PROMPTS = ""
    
    # 确保输出和日志目录存在
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)

    # 创建模型配置
    model_config = {
        "model_path": args.model_path,
        "use_api": args.use_api,
        "max_model_len": 16384,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt_mode": args.prompt_mode 
    }

    if args.use_api:
        model_config["api_type"] = args.api_type
    
    # 处理数据
    process_tablebench_data_batch(
        input_file=input_file, 
        output_file=args.output_file, 
        model_config=model_config, 
        log_file=args.log_file, 
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        start_from=args.start_from,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()