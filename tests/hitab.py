import json
import os
import logging
import time
import random
import sys
import argparse
import re
from datetime import datetime
from typing import Dict, Any, List, Tuple
from llm import initialize_client, call_api_with_retry
from prompt import COT_PROMPT_HITAB_TEMPLATE


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('hitab_processor')
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

def format_table(table_data):
    """Format table data as a string representation"""
    if not table_data or "texts" not in table_data:
        return ""

    rows = table_data["texts"]
    if not rows:
        return ""

    table_str = ""

    # If table has a title, add it to the table string
    if "title" in table_data and table_data["title"]:
        table_str += f"Title: {table_data['title']}\n\n"

    # Format table content
    for row in rows:
        row_str = " | ".join([str(cell) for cell in row])
        table_str += row_str + "\n"

    return table_str

def create_prompt_from_hitab(item: Dict[str, Any]) -> str:
    """Create prompt for HiTAB item"""
    # Get table, question, and context
    table_data = item.get("table", {})
    question = item.get("question", "")
    
    # Format table
    table_str = format_table(table_data)
    
    # Create prompt
    prompt = COT_PROMPT_HITAB_TEMPLATE.format(
        table=table_str,
        question=question
    )
    
    return prompt


def extract_final_answer(response):
    """Extract final answer from response, supporting multiple formats"""
    if not response:
        return ""
    
    # Try to extract answer from <answer> tags
    answer_tag_pattern = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_tag_pattern:
        answer_content = answer_tag_pattern.group(1).strip()
        
        # Check if there's an "Answer:" prefix inside the tags
        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[1].strip()
        return answer_content
    
    # Try to use regex to match "Answer: xxx" pattern
    answer_pattern = re.search(r'Answer:\s*(.*?)(?:$|\n|\.(?:\s|$))', response, re.IGNORECASE | re.DOTALL)
    if answer_pattern:
        return answer_pattern.group(1).strip()
    
    # If all extraction methods fail, return the original response
    return response

def check_answer_correctness(model_answer: Any, expected_answer: List) -> Tuple[bool, Any]:
    """
    Check if model answer is correct - simplified version
    Preprocess answers uniformly: lowercase, replace 'and' with commas, remove extra spaces
    """
    if model_answer is None:
        return False, None
    
    # Preprocessing function: convert to lowercase, replace 'and' with commas, normalize spaces
    def preprocess_answer(ans):
        if not isinstance(ans, str):
            return str(ans).lower().strip()
        
        # Convert to lowercase
        ans = ans.lower().strip()
        # Replace "and" with comma
        ans = re.sub(r'\s+and\s+', ', ', ans)
        # Normalize spaces
        ans = re.sub(r'\s+', ' ', ans)
        return ans
    
    # Preprocess expected answer (may be a list or single value)
    if isinstance(expected_answer, list):
        if len(expected_answer) == 1:
            # Single answer case
            expected_processed = preprocess_answer(expected_answer[0])
        else:
            # Multiple answers case, merge into comma-separated string
            expected_processed = ', '.join(preprocess_answer(item) for item in expected_answer)
    else:
        expected_processed = preprocess_answer(expected_answer)
    
    # Preprocess model answer
    model_processed = preprocess_answer(model_answer)
    
    # Check if model answer contains or equals expected answer
    # First try exact match
    if expected_processed == model_processed:
        return True, model_answer
    
    # Then try set matching (ignoring order)
    expected_items = set(item.strip() for item in expected_processed.split(',') if item.strip())
    model_items = set(item.strip() for item in model_processed.split(',') if item.strip())
    
    # If both sets are the same, consider it correct
    if expected_items and model_items and expected_items == model_items:
        return True, model_answer
    
    # Try numeric matching
    try:
        # If both are single numbers, perform exact matching
        expected_num = float(expected_processed.replace(',', ''))
        model_num = float(model_processed.replace(',', ''))
        
        # Exact match, no error allowed
        if expected_num == model_num:
            return True, model_answer
    except (ValueError, TypeError):
        pass
    
    # Other cases are considered incorrect
    return False, model_answer

def process_hitab_data(input_file, output_file, model_name, log_file, max_tokens=2048, temperature=0.7, start_from=0, api_port=8000):
    """Process HiTAB dataset"""
    logger = setup_logger(log_file)
    
    # Record start time
    start_time = time.time()
    logger.info(f"Started processing HiTAB data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"API port: {api_port}")
    logger.info(f"Starting from index: {start_from}")
    
    # Initialize model client
    try:
        client_info = initialize_client({"model_path": model_name, "api_port": api_port})
        logger.info(f"Model client initialized successfully, type: {client_info['model_type']}")
    except Exception as e:
        logger.error(f"Model client initialization failed: {e}")
        return
    
    # Read test data JSON file
    test_data = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # For JSONL format, read line by line
            for line in f:
                if line.strip():  # Skip empty lines
                    item = json.loads(line.strip())
                    test_data.append(item)
        logger.info(f"Loaded {len(test_data)} table data items from test file (JSONL format)")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
    # Check if intermediate results exist, if so, load them
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
    correct_count = 0
    
    # Process each data item
    for i, item in enumerate(test_data[start_from:], start=start_from):
        item_id = item.get("id", f"item_{i}")
        question = item.get("question", "")
        expected_answer = item.get("answer", [])
        
        logger.info(f"Processing item {i+1}/{len(test_data)}... [ID: {item_id}]")
        
        # Create prompt for HiTAB
        prompt = create_prompt_from_hitab(item)
        
        # Prepare user message
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Record API call start time
            call_start = time.time()
            
            # Call API
            api_result = call_api_with_retry(
                client_info=client_info,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.8,
                max_retries=10
            )
            
            # Initialize thinking variable to avoid undefined error
            thinking = None
            
            # Process API result - check if it's deepseek-r1 model
            if client_info["model_type"] in ["deepseek-r1", "deepseek-r1-inner"]:
                # deepseek-r1 returns three values: success flag, answer content and reasoning content
                success, answer, thinking = api_result
            else:
                # Other models return two values: success flag and answer content
                success, answer = api_result

            if not success:
                raise Exception(f"API call failed: {answer}")
            
            # Calculate API call time
            call_time = time.time() - call_start
            
            # Process returned response - extract token usage
            token_info = {}
            if client_info["model_type"] == "openai" and hasattr(answer, 'usage'):
                token_info = {
                    "completion_tokens": getattr(answer.usage, 'completion_tokens', 'N/A'),
                    "prompt_tokens": getattr(answer.usage, 'prompt_tokens', 'N/A'),
                    "total_tokens": getattr(answer.usage, 'total_tokens', 'N/A')
                }
                # Extract answer content
                answer = answer.choices[0].message.content
            else:
                token_info = {"note": "This model type does not provide token usage statistics"}
            
            # Try to extract final answer
            final_answer = extract_final_answer(answer)
            
            # Check if answer is correct
            is_correct, checked_answer = check_answer_correctness(final_answer, expected_answer)
            if is_correct:
                correct_count += 1
            
            # Build result object
            result = {
                "id": item_id,
                "question": question,
                "prompt": prompt,
                "model_answer": final_answer,
                "full_response": answer,
                "expected_answer": expected_answer,
                "is_correct": is_correct,
                "processing_time": call_time,
                "token_usage": token_info
            }
            
            # Add thinking content field (for deepseek-r1 model)
            if thinking is not None:
                result["reasoning"] = thinking
                logger.info(f"Model reasoning process: {thinking}")
            
            results.append(result)
            success_count += 1
            
            # Log detailed information
            logger.info(f"Question: {question}")
            logger.info(f"Expected answer: {expected_answer}")
            logger.info(f"Model answer: {final_answer}")
            logger.info(f"Is correct: {is_correct}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing item {i+1}: {e}")
            # Record error information
            result = {
                "id": item_id,
                "question": question,
                "expected_answer": expected_answer,
                "model_answer": f"Processing error: {str(e)}",
                "is_correct": False,
                "error": str(e)
            }
            results.append(result)
        
        # Save intermediate results every 5 items or when error occurs
        if (i + 1) % 5 == 0 or error_count > 0:
            try:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(test_data)})")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
    
    # Calculate accuracy
    accuracy = correct_count / len(test_data) if len(test_data) > 0 else 0
    
    # Save results to JSON file
    try:
        # Sort results by ID before saving
        results.sort(key=lambda x: x.get("id", ""))
        
        # Add evaluation metrics
        evaluation_metrics = {
            "total_questions": len(test_data),
            "correct_answers": correct_count,
            "accuracy": accuracy,
            "error_count": error_count
        }
        
        final_output = {
            "results": results,
            "metrics": evaluation_metrics
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_output, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")

    # Log summary information
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(test_data)}")
    logger.info(f"Processing failures: {error_count}/{len(test_data)}")
    logger.info(f"Correct answers: {correct_count}/{len(test_data)}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info("=" * 60)



def parse_arguments():
    parser = argparse.ArgumentParser(description='Process HiTAB dataset with LLM')
    
    parser.add_argument('--api_port', type=int, default=8000, help='API port for local model server')
    parser.add_argument('--output_file', type=str, help='Path to save results')
    parser.add_argument('--model_path', type=str, help='Model path or identifier')
    parser.add_argument('--log_file', type=str, help='Path to log file')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for model generation')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum tokens for model output')
    parser.add_argument('--start_from', type=int, default=0, help='Start processing from this index')
    parser.add_argument('--base_path', type=str, help='Base path for the project')
    
    return parser.parse_args()

def main():
    # 解析命令行参数
    args = parse_arguments()
    
    # 处理 base_path
    if args.base_path and os.path.exists(args.base_path):
        base_path = args.base_path
    
    if not base_path:
        print("Error: Unable to find project root directory")
        exit(1)
    
    print(f"Using root path: {base_path}")
    
    # 设置文件路径
    input_file = os.path.join(base_path, "data/HiTab/test.jsonl")
    
    # 使用命令行参数，如果提供了参数则使用参数值
    output_file = args.output_file
    model_name = args.model_path
    log_file = args.log_file
    max_tokens = args.max_tokens
    start_from = args.start_from
    temperature = args.temperature
    api_port = args.api_port
    
    # 确保输出目录和日志目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 处理数据
    process_hitab_data(input_file, output_file, model_name, log_file, max_tokens, temperature, start_from, api_port)

if __name__ == "__main__":
    main()