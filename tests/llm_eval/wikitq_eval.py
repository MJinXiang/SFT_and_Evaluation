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


def extract_predicted_answer(model_answer):
    """Extract predicted answer from model's response"""
    if not model_answer:
        return None
    
    # Try to match content wrapped in <answer> tags
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    answer_tag_match = re.search(answer_tag_pattern, model_answer, re.DOTALL)
    if answer_tag_match:
        model_answer = answer_tag_match.group(1).strip()
    
    # # Try to match content after "Answer:" or similar
    # match = re.search(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer, re.IGNORECASE)
    # if match:
    #     return match.group(1).strip()
    
    # If no specific answer format is found, return the whole response
    return model_answer.strip() if model_answer else None



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


def evaluate_with_llm(input_file, output_file, model_path, log_file=None, batch_size=8, 
                     tensor_parallel_size=1, max_model_len=8192, max_retries=5, verbose=True):
    """
    使用LLM评估WikiTableQuestions，支持本地VLLM和远程API调用
    """
    if log_file is None:
        # 仍保留自动生成逻辑，但确保目录存在
        dir_name = os.path.dirname(output_file)
        os.makedirs(dir_name, exist_ok=True)
        log_file = os.path.join(dir_name, f"llm_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    else:
        # 如果提供了日志文件路径，确保其父目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = setup_logger(log_file)
    logger.info(f"Starting LLM-based evaluation using model: {model_path}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    
    # Record start time
    start_time = time.time()
    
    # 判断使用本地VLLM还是API调用
    use_api = is_api_model(model_path)
    
    # Initialize appropriate evaluator
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
    
    # Load data
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, dict):
            data = [data]
        
        logger.info(f"Loaded {len(data)} samples for evaluation")
    except Exception as e:
        logger.error(f"Failed to load input file: {e}")
        return None

    # Prepare items for evaluation - 代码保持不变
    evaluation_items = []
    for i, item in enumerate(data):
        # Find expected answer
        expected_answer = None
        if "answer" in item:
            expected_answer = item["answer"]
        elif "truth_answer" in item:
            expected_answer = item["truth_answer"] 
        elif "true_answer" in item:
            expected_answer = item["true_answer"]
        elif "expected_answer" in item:
            expected_answer = item["expected_answer"]
        
        # 如果依然没有找到答案，使用默认值
        if expected_answer is None:
            expected_answer = "Unknown"
            logger.warning(f"Item {item.get('id', f'item-{i}')} missing golden answer, using default")
        
        # 首先检查是否已有提取的答案
        if "extracted_answer" in item:
            predicted_answer = item["extracted_answer"]
        else:
            # 使用强化的提取函数
            model_answer = item.get("model_answer", "")
            predicted_answer = extract_predicted_answer(model_answer)
        
        # 即使答案为空也继续评估
        evaluation_items.append({
            "id": item.get("id", f"item-{i}"),
            "question": item.get("question", ""),
            "expected_answer": expected_answer or "Unknown",
            "predicted_answer": predicted_answer or "No answer",
            "full_response": item.get("model_answer", "")
        })

    
    logger.info(f"Prepared {len(evaluation_items)} items for LLM evaluation")
    
    # 对于API评估器，batch_size表示并行请求数量
    if use_api:
        logger.info(f"Using API evaluator with concurrency={batch_size}")
    else:
        logger.info(f"Using VLLM evaluator with batch_size={batch_size}")
        
    evaluation_results = evaluator.batch_evaluate(evaluation_items, batch_size=batch_size)
    
    # 后续处理和统计部分不变...
    total_evaluated = len(evaluation_results)
    correct_count = sum(1 for result in evaluation_results if result["is_correct"])
    accuracy = correct_count / total_evaluated if total_evaluated > 0 else 0
    
    stats = {
        "total_samples": len(data),
        "evaluated_samples": total_evaluated,
        "correct_count": correct_count,
        "accuracy": accuracy,
        "model_used": model_path,
        "evaluation_time": time.time() - start_time
    }
    
    final_result = {
        "summary": stats,
        "results": evaluation_results
    }

    # 保存结果
    logger.info(f"Evaluation complete. Saving results to {output_file}...")
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, indent=4, ensure_ascii=False)
        logger.info(f"Evaluation results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving evaluation results: {e}")
    
    # 输出摘要
    if verbose:
        logger.info("\n===== LLM Evaluation Summary =====")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Evaluated samples: {stats['evaluated_samples']}")
        logger.info(f"Correct answers: {stats['correct_count']}")
        logger.info(f"Accuracy: {stats['accuracy'] * 100:.2f}%")
        logger.info(f"Total evaluation time: {stats['evaluation_time']:.2f} seconds")
        logger.info(f"Evaluation model: {model_path} (via {'API' if use_api else 'VLLM'})")
    
    return final_result


def parse_arguments():
    parser = argparse.ArgumentParser(description="LLM-based WikiTQ prediction evaluation")
    
    parser.add_argument("--results_file", 
                       help="Input file path with predictions")
    parser.add_argument("--output_file",
                       help="Output file path for evaluation results")
    parser.add_argument("--model_path", 
                       help="Path to the LLM model for evaluation")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Path for log file (optional)")
    parser.add_argument("--batch_size", type=int, default=8, 
                       help="Batch size for LLM evaluation")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                       help="Tensor parallelism size")
    parser.add_argument("--base_path", type=str, default=None,
                       help="Base path for the project (optional)")
    parser.add_argument("--max_retries", type=int, default=10,
                       help="Maximum retry attempts for API calls (default: 5)")
    
    return parser.parse_args()


def main():
    """Main function that handles command line arguments"""
    args = parse_arguments()
    
    # Process base_path if provided
    if args.base_path:
        if not os.path.exists(args.base_path):
            print(f"Warning: Provided base_path {args.base_path} does not exist")
    
    # Run LLM-based evaluation
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
    
    if result is None:
        print("Evaluation failed")
        exit(1)


if __name__ == "__main__":
    main()