import json
import re
import os
import time
import logging
import sys
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompt import WIKITQ_EVAL

import os.path

# 获取当前文件所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 向上一级获取tests目录
project_root = os.path.dirname(current_dir)

sys.path.append(project_root)
sys.path.append(project_root)
from utils.llm import call_api_with_retry, initialize_client
from eval.wikitq_eval import evaluate_answers, normalize_answer, exact_match_enhanced, extract_predicted_answer


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('wikitq_evaluator')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


class BaseEvaluator:
    """基础评估器类，定义共享的方法"""
    
    def __init__(self, model_path, logger=None):
        self.model_path = model_path
        self.logger = logger or logging.getLogger('wikitq_evaluator')
    
    def create_evaluation_prompt(self, question, reference_answer, predicted_answer):
        """
        创建评估提示
        """
        prompt = WIKITQ_EVAL.format(
            question=question,
            candidate_answer=predicted_answer,
            correct_answer=reference_answer
        )
        return prompt


class VLLMEvaluator(BaseEvaluator):
    """  
    使用VLLM执行本地模型评估
    """  
    
    def __init__(self, model_path, max_model_len=8192, tensor_parallel_size=1, logger=None):  
        super().__init__(model_path, logger)
        
        # Default EOS tokens list
        self.EOS = ["<|im_end|>", "</s>"]
        
        self.logger.info(f"Initializing VLLM evaluator with model: {model_path}")
        
        try:
            self.model = LLM(  
                model=model_path,  
                max_model_len=max_model_len,  
                trust_remote_code=True,  
                distributed_executor_backend='ray',  
                tensor_parallel_size=tensor_parallel_size
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.logger.info("VLLM evaluator initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize VLLM: {e}")
            raise

    def batch_evaluate(self, evaluation_items, batch_size=16):
        """
        使用VLLM批量评估样本
        """
        all_results = []
        
        # Create batches
        for i in range(0, len(evaluation_items), batch_size):
            batch = evaluation_items[i:i+batch_size]
            
            # Create prompts for this batch
            prompts = []
            for item in batch:
                prompt = self.create_evaluation_prompt(
                    item["question"], 
                    item["expected_answer"],
                    item["predicted_answer"]
                )
                prompts.append(prompt)
            
            # Convert prompts to chat format
            chat_prompts = []
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                chat_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                chat_prompts.append(chat_prompt)
            
            # Generate evaluations in batch
            try:
                batch_start_time = time.time()
                self.logger.info(f"Evaluating batch of {len(batch)} items using VLLM")
                
                responses = self.model.generate(
                    prompts=chat_prompts,
                    sampling_params=SamplingParams(
                        max_tokens=1024,
                        temperature=0.0,
                        top_p=1.0,
                        stop=self.EOS,
                    ),
                    use_tqdm=True,
                )
                
                # Process responses
                for j, (item, response) in enumerate(zip(batch, responses)):
                    evaluation_text = response.outputs[0].text.strip()
                    
                    is_correct, answer_text = extract_evaluation_result(evaluation_text)
                    
                    result = {
                        "id": item.get("id", f"item-{i+j}"),
                        "question": item["question"],
                        "expected_answer": item["expected_answer"],
                        "predicted_answer": item["predicted_answer"],
                        "is_correct": is_correct,
                        "explanation": evaluation_text,
                    }
                    all_results.append(result)

                    # 添加详细的每个样例日志
                    self.logger.info(f"Item {i+j+1}: ID={result['id']}")
                    self.logger.info(f"  Question: {result['question']}")
                    self.logger.info(f"  Expected: {result['expected_answer']}")
                    self.logger.info(f"  Predicted: {result['predicted_answer']}")
                    self.logger.info(f"  LLM Decision: {answer_text}")
                    self.logger.info(f"  Marked as: {'Correct' if is_correct else 'Incorrect'}")
                    self.logger.info("  " + "-"*50)
                
                batch_time = time.time() - batch_start_time
                self.logger.info(f"Batch evaluated in {batch_time:.2f}s ({batch_time/len(batch):.2f}s per item)")
                
            except Exception as e:
                self.logger.error(f"Error in batch evaluation: {e}")
                
                # Create empty results for failed items
                for item in batch:
                    result = {
                        "id": item.get("id", "unknown"),
                        "question": item["question"],
                        "expected_answer": item["expected_answer"],
                        "predicted_answer": item["predicted_answer"],
                        "is_correct": False,
                        "explanation": f"Evaluation failed: {str(e)}",
                    }
                    all_results.append(result)
        
        return all_results


class APIEvaluator(BaseEvaluator):
    """
    使用API调用闭源模型执行评估，支持并行处理
    """
    
    def __init__(self, model_path, logger=None, max_retries=5):
        """
        初始化API评估器
        
        Args:
            model_path: API模型路径或名称
            logger: 日志记录器
            max_retries: 最大重试次数
        """
        super().__init__(model_path, logger)
        self.max_retries = max_retries
        self.logger.info(f"Initializing API evaluator for model: {model_path} with max_retries={max_retries}")
        
        try:
            self.client_info = initialize_client({"model_path": model_path})
            self.logger.info(f"API client initialized for {model_path}")
        except Exception as e:
            self.logger.error(f"Failed to initialize API client: {e}")
            raise
    
    def evaluate_single_item(self, item, item_index):
        """
        评估单个样本，便于并行处理
        
        Args:
            item: 要评估的样本
            item_index: 样本索引
            
        Returns:
            评估结果字典
        """
        item_start_time = time.time()
        
        try:
            prompt = self.create_evaluation_prompt(
                item["question"], 
                item["expected_answer"],
                item["predicted_answer"]
            )
            
            # 准备API调用
            messages = [{"role": "user", "content": prompt}]
            
            # 调用API，使用类中设定的最大重试次数
            success, response = call_api_with_retry(
                self.client_info,
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
                top_p=1.0,
                max_retries=self.max_retries  # 使用实例变量控制最大重试次数
            )
            
            if success:
                # 根据模型类型处理响应
                if self.client_info["model_type"] == "openai":
                    evaluation_text = response.choices[0].message.content
                else:
                    # 闭源模型直接返回文本内容
                    evaluation_text = response
                
                is_correct, answer_text = extract_evaluation_result(evaluation_text)
                
                result = {
                    "id": item.get("id", f"item-{item_index}"),
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "predicted_answer": item["predicted_answer"],
                    "is_correct": is_correct,
                    "explanation": evaluation_text,
                    "answer_text": answer_text,  # 保存以便记录日志
                    "success": True,
                    "processing_time": time.time() - item_start_time
                }
            else:
                # API调用失败
                self.logger.error(f"API call failed for item {item.get('id', f'item-{item_index}')} after {self.max_retries} retries: {response}")
                result = {
                    "id": item.get("id", f"item-{item_index}"),
                    "question": item["question"],
                    "expected_answer": item["expected_answer"],
                    "predicted_answer": item["predicted_answer"],
                    "is_correct": False,
                    "explanation": f"API call failed after {self.max_retries} retries: {response}",
                    "success": False,
                    "processing_time": time.time() - item_start_time
                }
                
        except Exception as e:
            self.logger.error(f"Error evaluating item {item_index}: {e}")
            result = {
                "id": item.get("id", f"item-{item_index}"),
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "predicted_answer": item["predicted_answer"],
                "is_correct": False,
                "explanation": f"Evaluation error: {str(e)}",
                "success": False,
                "processing_time": time.time() - item_start_time
            }
            
        return result
    
    def batch_evaluate(self, evaluation_items, batch_size=16):
        """
        使用API并行调用批量评估样本
        
        Args:
            evaluation_items: 要评估的样本列表
            batch_size: 并行处理的批次大小
            
        Returns:
            评估结果列表
        """
        import concurrent.futures
        
        all_results = []
        total_items = len(evaluation_items)
        
        self.logger.info(f"Starting parallel API evaluation of {total_items} items with concurrency={batch_size}")
        
        # 使用ThreadPoolExecutor进行并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as executor:
            # 创建未来任务列表
            future_to_item = {
                executor.submit(self.evaluate_single_item, item, i): (i, item) 
                for i, item in enumerate(evaluation_items)
            }
            
            # 处理完成的任务
            progress_counter = 0
            batch_start_time = time.time()
            
            for future in concurrent.futures.as_completed(future_to_item):
                i, item = future_to_item[future]
                progress_counter += 1
                
                try:
                    result = future.result()
                    all_results.append(result)
                    
                    # 记录评估结果的日志
                    self.logger.info(f"Item {progress_counter}/{total_items}: ID={result['id']}")
                    self.logger.info(f"  Question: {result['question']}")
                    self.logger.info(f"  Expected: {result['expected_answer']}")
                    self.logger.info(f"  Predicted: {result['predicted_answer']}")
                    
                    if result["success"]:
                        self.logger.info(f"  LLM Decision: {result.get('answer_text', 'Unknown')}")
                    else:
                        self.logger.info(f"  LLM Decision: Failed")
                        
                    self.logger.info(f"  Marked as: {'Correct' if result['is_correct'] else 'Incorrect'}")
                    self.logger.info(f"  Processing time: {result['processing_time']:.2f}s")
                    self.logger.info("  " + "-"*50)
                    
                    # 每10个样本或最后一个样本时输出进度
                    if progress_counter % 10 == 0 or progress_counter == total_items:
                        elapsed_time = time.time() - batch_start_time
                        self.logger.info(f"Progress: {progress_counter}/{total_items} items processed, "\
                                         f"elapsed time: {elapsed_time:.2f}s, "\
                                         f"average: {elapsed_time/progress_counter:.2f}s per item")
                        
                except Exception as e:
                    self.logger.error(f"Unexpected error processing result for item {i}: {e}")
        
        elapsed_time = time.time() - batch_start_time
        self.logger.info(f"Completed API evaluation of {total_items} items in {elapsed_time:.2f}s "\
                         f"({elapsed_time/total_items:.2f}s per item)")
        
        # 按照原始顺序排序结果
        all_results.sort(key=lambda x: int(x["id"].split("-")[-1]) if "-" in x["id"] else 0)
        
        return all_results


def extract_evaluation_result(evaluation_text):
    """
    从LLM评估文本中提取Yes/No结果
    
    Args:
        evaluation_text: LLM生成的评估文本
        
    Returns:
        bool: 提取的结果，True表示回答正确，False表示回答错误
        str: 提取到的原始答案文本（用于日志）
    """
    yes_match = re.search(r'\bYes\b', evaluation_text)
    no_match = re.search(r'\bNo\b', evaluation_text)
    
    # 如果找到明确的Yes，则返回True
    if yes_match and not no_match:
        return True, "Yes"
    
    # 如果找到明确的No，则返回False
    if no_match and not yes_match:
        return False, "No"
    
    # 如果同时有Yes和No或者都没有，检查最后一行或者最后一个非空字符串
    lines = [line.strip() for line in evaluation_text.split('\n') if line.strip()]
    if lines:
        last_line = lines[-1].strip()
        if last_line.lower() == "yes":
            return True, "Yes (from last line)"
        elif last_line.lower() == "no":
            return False, "No (from last line)"
    
    # 检查文本中是否包含"Yes"或"No"的频率，以防有干扰
    yes_count = len(re.findall(r'\byes\b', evaluation_text.lower()))
    no_count = len(re.findall(r'\bno\b', evaluation_text.lower()))
    
    if yes_count > no_count:
        return True, f"Yes (inferred from {yes_count} occurrences vs {no_count} no's)"
    elif no_count > yes_count:
        return False, f"No (inferred from {no_count} occurrences vs {yes_count} yes's)"
    
    # 无法确定结果时，默认为False
    return False, "Unknown (defaulting to No)"


def is_api_model(model_path):
    """
    判断是否是需要通过API调用的模型
    """
    # 检查模型路径是本地路径还是API模型名称
    api_model_prefixes = ["gemini", "claude", "claude-3-7-sonnet", "deepseek-r1"]
    
    # 如果是本地路径(包含斜杠或反斜杠)，则使用VLLM
    if "/" in model_path or "\\" in model_path:
        return False
        
    # 检查是否以已知API模型前缀开头
    for prefix in api_model_prefixes:
        if model_path.startswith(prefix):
            return True
    
    # 默认使用VLLM
    return False


def combined_evaluate(input_file, output_file, model_path, log_file=None, batch_size=8,
                     tensor_parallel_size=1, max_model_len=8192, max_retries=5, verbose=True):
    """
    结合精确匹配和LLM评估的混合评估方法：
    1. 首先使用精确匹配方法评估所有样本
    2. 对精确匹配判定为错误的样本，使用LLM进行二次评估
    3. 融合两种评估结果
    """
    if log_file is None:
        dir_name = os.path.dirname(output_file)
        os.makedirs(dir_name, exist_ok=True)
        log_file = os.path.join(dir_name, f"combined_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logger = setup_logger(log_file)
    logger.info(f"Starting combined evaluation (exact match + LLM) using model: {model_path}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 第一步：使用精确匹配方法进行评估
    logger.info("Step 1: Performing exact match evaluation...")
    
    # 先创建临时输出文件路径用于精确匹配评估
    temp_exact_match_file = os.path.join(os.path.dirname(output_file), 
                                        f"temp_exact_match_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    exact_match_results = evaluate_answers(input_file, temp_exact_match_file, verbose=False)
    
    if exact_match_results is None:
        logger.error("Exact match evaluation failed")
        return None
    
    # 收集精确匹配评估结果的统计信息
    em_stats = exact_match_results["summary"]
    em_results_list = exact_match_results["results"]
    
    # 创建索引便于快速查找
    em_results_by_id = {item["id"]: item for item in em_results_list}
    
    logger.info(f"Exact match evaluation completed: {em_stats['exact_matches']} / {em_stats['answered_samples']} correct")
    
    # 第二步：筛选出精确匹配评估为错误的样本，为LLM评估准备数据
    items_for_llm = []
    for item in em_results_list:
        if item.get("is_exact_match") is False:  # 精确匹配判断为错误
            items_for_llm.append({
                "id": item["id"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "predicted_answer": item["predicted_answer"],
            })
    
    logger.info(f"Step 2: Selected {len(items_for_llm)} samples that failed exact match for LLM evaluation")
    
    # 如果没有需要LLM评估的样本，直接返回精确匹配结果
    if not items_for_llm:
        logger.info("No samples require LLM evaluation, using exact match results only")
        
        # 将精确匹配结果直接写入指定的输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(exact_match_results, f, indent=4, ensure_ascii=False)
        
        # 删除临时文件
        try:
            os.remove(temp_exact_match_file)
        except:
            pass
            
        if verbose:
            logger.info("\n===== Combined Evaluation Summary =====")
            logger.info(f"Total samples: {em_stats['total_samples']}")
            logger.info(f"Exact match correct: {em_stats['exact_matches']}")
            logger.info(f"Exact match rate: {em_stats['exact_match_rate'] * 100:.2f}%")
        return exact_match_results
    
    # 第三步：保存需要LLM评估的样本到临时文件
    temp_file = os.path.join(os.path.dirname(output_file), f"temp_for_llm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(temp_file, 'w', encoding='utf-8') as f:
        json.dump(items_for_llm, f, indent=4, ensure_ascii=False)
    
    # 第四步：使用LLM对这些样本进行评估
    logger.info("Step 3: Performing LLM evaluation on samples that failed exact match...")
    llm_output_file = os.path.join(os.path.dirname(output_file), f"llm_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # 判断是否使用API模型
    use_api = is_api_model(model_path)
    
    # 初始化相应的评估器
    try:
        if use_api:
            logger.info(f"Using API evaluator for model: {model_path} with max_retries={max_retries}")
            evaluator = APIEvaluator(model_path=model_path, logger=logger, max_retries=max_retries)
        else:
            logger.info(f"Using local VLLM evaluator for model: {model_path}")
            evaluator = VLLMEvaluator(
                model_path=model_path,
                max_model_len=max_model_len,
                tensor_parallel_size=tensor_parallel_size,
                logger=logger
            )
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return None
    
    # 进行LLM评估
    logger.info(f"Evaluating {len(items_for_llm)} items with LLM")
    llm_evaluation_results = evaluator.batch_evaluate(items_for_llm, batch_size=batch_size)
    
    # 第五步：合并两种评估结果
    logger.info("Step 4: Merging exact match and LLM evaluation results...")
    
    # 将LLM评估结果按ID整理成字典便于查找
    llm_results_by_id = {item["id"]: item for item in llm_evaluation_results}
    
    # 创建最终的结合结果列表
    combined_results_list = []
    
    # 统计数字
    combined_correct = 0
    total_evaluated = 0
    
    # 处理每个样本
    for item in em_results_list:
        item_id = item["id"]
        combined_item = dict(item)  # 复制原始项
        
        # 检查该样本是否有精确匹配结果
        if item.get("is_exact_match") is True:
            # 精确匹配正确，无需LLM评估
            combined_item["evaluation_method"] = "exact_match"
            combined_item["is_correct"] = True
            combined_correct += 1
        elif item_id in llm_results_by_id:
            # 精确匹配错误，使用LLM评估结果
            llm_result = llm_results_by_id[item_id]
            combined_item["evaluation_method"] = "llm"
            combined_item["is_correct"] = llm_result["is_correct"]
            combined_item["llm_explanation"] = llm_result["explanation"]
            if llm_result["is_correct"]:
                combined_correct += 1
        else:
            # 该样本没有LLM评估结果（可能是因为预期答案缺失）
            combined_item["evaluation_method"] = "exact_match"
            combined_item["is_correct"] = False
        
        total_evaluated += 1
        combined_results_list.append(combined_item)
    
    # 计算准确率
    accuracy = combined_correct / total_evaluated if total_evaluated > 0 else 0
    
    # 创建最终结果
    combined_stats = {
        "total_samples": em_stats["total_samples"],
        "evaluated_samples": total_evaluated,
        "exact_match_correct": em_stats["exact_matches"],
        "llm_additional_correct": combined_correct - em_stats["exact_matches"],
        "total_correct": combined_correct,
        "accuracy": accuracy,
        "evaluation_model": model_path,
        "evaluation_time": time.time() - start_time
    }
    
    final_result = {
        "summary": combined_stats,
        "results": combined_results_list
    }
    
    # 保存结果
    logger.info(f"Saving combined evaluation results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, indent=4, ensure_ascii=False)
    
    # 清理临时文件
    try:
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(llm_output_file):
            os.remove(llm_output_file)
        if os.path.exists(temp_exact_match_file):
            os.remove(temp_exact_match_file)
    except Exception as e:
        logger.warning(f"Could not clean up temporary files: {e}")
    
    # 输出摘要
    if verbose:
        logger.info("\n===== Combined Evaluation Summary =====")
        logger.info(f"Total samples: {combined_stats['total_samples']}")
        logger.info(f"Exact match correct: {combined_stats['exact_match_correct']}")
        logger.info(f"Additional LLM correct: {combined_stats['llm_additional_correct']}")
        logger.info(f"Total correct: {combined_stats['total_correct']}")
        logger.info(f"Final accuracy: {combined_stats['accuracy'] * 100:.2f}%")
        logger.info(f"Total evaluation time: {combined_stats['evaluation_time']:.2f} seconds")
    
    return final_result


def parse_arguments():
    parser = argparse.ArgumentParser(description="Combined exact match + LLM evaluation for WikiTQ")
    
    parser.add_argument("--results_file", required=True, 
                       help="Input file path with predictions")
    parser.add_argument("--output_file", required=True,
                       help="Output file path for evaluation results")
    parser.add_argument("--model_path", required=True,
                       help="Path to the LLM model for evaluation")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Path for log file (optional)")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for LLM evaluation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallelism size")
    parser.add_argument("--max_retries", type=int, default=5,
                       help="Maximum retry attempts for API calls")
    parser.add_argument("--evaluation_mode", type=str, default="combined", 
                       choices=["exact", "llm", "combined"],
                       help="Evaluation mode: exact match only, LLM only, or combined")
    parser.add_argument("--base_path", type=str, default=None,
                       help="Base path for the project (optional)")
    
    return parser.parse_args()


def main():
    """Main function that handles command line arguments"""
    args = parse_arguments()
    
    # Process base_path if provided
    if args.base_path:
        if not os.path.exists(args.base_path):
            print(f"Warning: Provided base_path {args.base_path} does not exist")
    
    # 根据评估模式选择评估方法
    if args.evaluation_mode == "exact":
        # 仅使用精确匹配评估
        result = evaluate_answers(
            input_file=args.results_file,
            output_file=args.output_file,
            verbose=True
        )
    elif args.evaluation_mode == "llm":
        # 仅使用LLM评估
        from wikitq_eval import evaluate_with_llm
        result = evaluate_with_llm(
            input_file=args.results_file,
            output_file=args.output_file,
            model_path=args.model_path,
            log_file=args.log_file,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            max_retries=args.max_retries,
            verbose=True
        )
    else:  # combined (default)
        # 使用组合评估
        result = combined_evaluate(
            input_file=args.results_file,
            output_file=args.output_file,
            model_path=args.model_path,
            log_file=args.log_file,
            batch_size=args.batch_size,
            tensor_parallel_size=args.tensor_parallel_size,
            max_retries=args.max_retries,
            verbose=True
        )
    
    if result is None:
        print("Evaluation failed")
        exit(1)


if __name__ == "__main__":
    main()