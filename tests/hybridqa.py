#### python
# filepath: /netcache/mengjinxiang/Project/LLaMA-Factory-main/baseline/hybridqa_test.py

import json
import os
import logging
import time
import random
import sys
import re
import string
from datetime import datetime
from llm import initialize_client, call_api_with_retry
from prompt import COT_PROMPT_HYBRIDQA_TEMPLATE

# 设置日志
def setup_logger(log_file):
    """设置日志记录器"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('hybridqa_test_processor')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def format_table_for_prompt(table_data):
    """将表格格式化为prompt中易于阅读的格式"""
    result = []
    
    # 添加介绍信息
    if 'intro' in table_data and table_data['intro']:
        result.append(f"Introduction: {table_data['intro']}")
    
    # 添加章节标题和文本
    if 'section_title' in table_data and table_data['section_title']:
        section_text = table_data.get('section_text', '')
        result.append(f"Section Description: {section_text}")
    
    # 添加表格内容
    if 'header' in table_data and table_data['header'] and 'data' in table_data:
        # 获取表头
        if table_data['header'][0]:
            headers = table_data['header'][0]
            # 格式化表头
            header_row = " | ".join(headers)
            result.append(header_row)
        
        # 格式化数据行
        for row in table_data['data']:
            result.append(" | ".join(row))
    
    return "\n".join(result)

def format_text_for_prompt(link_data):
    """将链接文本格式化为prompt中的字符串格式，添加文本前缀"""
    result = []
    
    for link, content in link_data.items():
        # 提取链接的最后部分作为文本前缀
        link_parts = link.split('/')
        prefix = link_parts[-1] if link_parts else link
        
        # 添加带前缀的文本
        result.append(f"{prefix}: {content}")
    
    return "\n\n".join(result)

def create_prompt_from_hybridqa(item):
    """为HybridQA数据创建提示"""
    # 格式化表格
    table_str = format_table_for_prompt(item["table"])
    
    # 格式化文本
    text_str = format_text_for_prompt(item["relevant_links"])
    
    # 生成提示
    prompt = COT_PROMPT_HYBRIDQA_TEMPLATE.format(
        table=table_str, 
        text=text_str, 
        question=item["question"]
    )
    
    return prompt

# def extract_answer_from_response(model_answer):
#     """从模型回答中提取最终答案"""
#     # 寻找"Answer: "模式
#     match = re.search(r'Answer:\s*(.*?)$', model_answer, re.DOTALL)
#     if match:
#         return match.group(1).strip()
#     return ""

def extract_answer_from_response(model_answer):
    """
    从模型回答中提取最终答案，支持多种格式：
    1. "Answer: xxx"格式
    2. "<answer>xxx</answer>"标签格式
    3. "<answer>Answer: xxx</answer>"组合格式
    """
    # 如果响应为空，返回空字符串
    if not model_answer:
        return ""
    
    # 检查是否有<answer>标签
    answer_tag_pattern = re.search(r'<answer>(.*?)</answer>', model_answer, re.DOTALL)
    if answer_tag_pattern:
        # 从<answer>标签中提取内容
        answer_content = answer_tag_pattern.group(1).strip()
        
        # 检查answer标签内是否有"Answer:"标记
        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[1].strip()
        # 否则直接返回标签内的全部内容
        return answer_content
    
    # 如果没有<answer>标签但有"Answer:"标记
    elif "Answer:" in model_answer:
        return model_answer.split("Answer:")[1].strip()
    
    # 如果没有任何标记，返回整个响应
    return model_answer.strip()

def process_hybridqa_test_data(input_file, output_file, model_name, log_file, max_tokens=1024, temperature=0.0, start_from=0):
    """处理HybridQA测试数据集，无需评估答案正确性"""
    logger = setup_logger(log_file)
    
    # 记录开始时间
    start_time = time.time()
    logger.info(f"Started processing HybridQA test data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Starting from index: {start_from}")
    
    # 初始化模型客户端
    try:
        client_info = initialize_client({"model_path": model_name})
        model_type = client_info["model_type"]
        logger.info(f"Model client initialized successfully, type: {model_type}")
    except Exception as e:
        logger.error(f"Model client initialization failed: {e}")
        return
    
    # 读取JSON文件
    data_items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_items = json.load(f)
        logger.info(f"Loaded {len(data_items)} questions from input file")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
    # 检查是否存在中间结果，如果有则加载
    results = []
    if start_from > 0 and os.path.exists(f"{output_file}.temp"):
        try:
            with open(f"{output_file}.temp", 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded intermediate results with {len(results)} records")
        except Exception as e:
            logger.error(f"Failed to load intermediate results: {e}, starting from beginning")
            start_from = 0
    
    success_count = len(results)
    error_count = 0
    
    # 准备评估结果格式
    predictions = {}
    
    # 处理每个数据项
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("question_id", f"item-{i}")
        
        logger.info(f"Processing item {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        # 获取问题
        question = item.get("question", "")
        
        # 创建提示
        prompt = create_prompt_from_hybridqa(item)
        
        # 准备用户消息
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # 记录API调用开始时间
            call_start = time.time()
            
            # 调用API
            api_result = call_api_with_retry(
                client_info=client_info,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                max_retries=10
            )
            
            # 初始化thinking变量以避免未定义错误
            thinking = None
            
            # 处理API结果 - 检查是否是deepseek-r1模型
            if model_type in ["deepseek-r1", "deepseek-r1-inner"]:
                # deepseek-r1返回三个值：成功标志、答案内容和推理内容
                success, answer, thinking = api_result
            else:
                # 其他模型返回两个值：成功标志和答案内容
                success, answer = api_result

            if not success:
                raise Exception(f"API call failed: {answer}")
            
            # 计算API调用时间
            call_time = time.time() - call_start
            
            # 处理返回的响应 - 提取token使用情况
            token_info = {}
            if model_type == "openai" and hasattr(answer, 'usage'):
                token_info = {
                    "completion_tokens": getattr(answer.usage, 'completion_tokens', 'N/A'),
                    "prompt_tokens": getattr(answer.usage, 'prompt_tokens', 'N/A'),
                    "total_tokens": getattr(answer.usage, 'total_tokens', 'N/A')
                }
                # 提取答案内容
                answer = answer.choices[0].message.content
            else:
                token_info = {"note": "This model type does not provide token usage statistics"}
            
            # 提取最终答案
            extracted_answer = extract_answer_from_response(answer)
            
            # 构建结果对象
            result = {
                "question_id": item_id,
                "question": question,
                "model_answer": answer,
                "extracted_answer": extracted_answer,
                "processing_time": call_time,
                "token_usage": token_info
            }
            
            # 添加思考内容字段（适用于deepseek-r1模型）
            if thinking is not None:
                result["reasoning"] = thinking
            
            results.append(result)
            success_count += 1
            
            # 添加到预测字典中，用于生成评估格式
            predictions[item_id] = extracted_answer
            
            # 记录详细信息
            logger.info(f"Question: {question}")
            logger.info(f"Model answer: {extracted_answer}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing item {i+1}: {e}")
            # 记录错误信息
            result = {
                "question_id": item_id,
                "question": question,
                "model_answer": f"Processing error: {str(e)}",
                "error": str(e)
            }
            results.append(result)
            
            # 添加空预测以避免评估错误
            predictions[item_id] = ""
        
        # 每5个项目或出现错误时保存中间结果
        if (i + 1) % 5 == 0 or error_count > 0:
            try:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(data_items)})")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
    
    # 保存详细结果到JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save detailed results file: {e}")
    
    # 保存评估格式的输出（只包含问题ID和预测答案）
    eval_output_file = output_file.replace('.json', '_eval.json')
    try:
        # 为评估格式创建数据结构
        eval_data = [
            {
                "question_id": qid,
                "pred": pred
            }
            for qid, pred in predictions.items()
        ]
        
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation format results saved to {eval_output_file}")
    except Exception as e:
        logger.error(f"Failed to save evaluation format results: {e}")

    # 记录摘要信息
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Processing failures: {error_count}/{len(data_items)}")
    logger.info("=" * 60)

def main():
    # 检测正确的根路径
    base_paths = [
        "/mnt/usercache/mengjinxiang/Project/LLaMA-Factory-main",
        # "/netcache/mengjinxiang/Project/LLaMA-Factory-main",
    ]
    
    base_path = None
    for path in base_paths:
        if os.path.exists(path):
            base_path = path
            break
    
    if not base_path:
        print("Error: Unable to find the project root directory")
        exit(1)
    
    print(f"Using root path: {base_path}")
    
    # 参数设置
    input_file = os.path.join(base_path, "data/hybridqa/test_top_1.json")  # 处理好的测试集
    output_file = os.path.join(base_path, "results/hybridqa/hybridqa_sft_ppo_results.json")
    model_name = "/mnt/usercache/mengjinxiang/Project/TinyZero/checkpoints/TinyZero/hybridqa-sft-ppo-qwen2.5-3b-instruct/actor/global_step_800"  # 可以使用"deepseek-r1"或本地模型路径   /mnt/usercache/huggingface/Qwen2.5-3B-Instruct   /mnt/usercache/mengjinxiang/Project/TinyZero/checkpoints/TinyZero/hybridqa-qwen2.5-3b-instruct/actor/global_step_400  /mnt/usercache/huggingface/Qwen2.5-3B-Instruct
    log_file = os.path.join(base_path, "results/hybridqa/logs/hybridqa_sft_ppo_processing.log")
    max_tokens = 2048  # 最大输出token数
    temperature = 0.0  
    
    # 检查点参数 - 从哪个数据项开始（0表示从头开始）
    start_from = 0
    
    # 确保输出目录和日志目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 处理数据
    process_hybridqa_test_data(input_file, output_file, model_name, log_file, max_tokens, temperature, start_from)

if __name__ == "__main__":
    main()