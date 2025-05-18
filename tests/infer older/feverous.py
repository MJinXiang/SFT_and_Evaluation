import json
import os
import logging
import time
import re
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from llm import initialize_client, call_api_with_retry
from prompt import COT_PROMPT_FEVEROUS_TEMPLATE


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('feverous_processor')
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
    """Format table data into string representation"""
    if not table_data:
        return ""
    
    result = ""
    for table_id, table_content in table_data.items():
        result += f"Table {table_id}:\n"
        table = table_content.get("table", [])
        for row in table:
            result += " | ".join([str(cell) for cell in row]) + "\n"
        result += "\n"
    
    return result.strip()

def format_sentences(sentences_data):
    """Format sentence data into string representation"""
    if not sentences_data:
        return ""
    
    result = "Text excerpts:\n"
    for key, sentence in sentences_data.items():
        # Clean Wiki link format [[link|text]] or [[link]]
        # clean_sentence = re.sub(r'\[\[([^\]|]+)\|([^\]]+)\]\]', r'\2', sentence)
        # clean_sentence = re.sub(r'\[\[([^\]]+)\]\]', r'\1', clean_sentence)
        result += f"- {sentence}\n"
    
    return result

def extract_answer_from_response(model_answer: str) -> Optional[str]:
    """Extract final answer from model response, supporting multiple formats"""
    if not model_answer:
        return None
    
    # Try to extract answer from <answer> tags
    answer_tag_pattern = re.search(r'<answer>(.*?)</answer>', model_answer, re.DOTALL)
    if answer_tag_pattern:
        answer_content = answer_tag_pattern.group(1).strip()
        
        # Check if there's an "Answer:" prefix inside the tags
        if "Answer:" in answer_content:
            after_answer = answer_content.split("Answer:", 1)[1].strip()
        else:
            after_answer = answer_content
        
        # Check for label in the extracted content
        if re.search(r'(SUPPORTS|REFUTES|NOT ENOUGH INFO)', after_answer, re.IGNORECASE):
            answer_text = after_answer.upper()
            if "SUPPORTS" in answer_text:
                return "SUPPORTS"
            elif "REFUTES" in answer_text:
                return "REFUTES"
            elif "NOT ENOUGH INFO" in answer_text:
                return "NOT ENOUGH INFO"
    
    # Try to use regex to match "Answer: xxx" pattern
    match = re.search(r'Answer:\s*(SUPPORTS|REFUTES|NOT ENOUGH INFO|supports|refutes|not enough info)', 
                      model_answer, re.IGNORECASE | re.MULTILINE)
    if match:
        answer_text = match.group(1).strip().upper()
        # Normalize answer format
        if "SUPPORTS" in answer_text:
            return "SUPPORTS"
        elif "REFUTES" in answer_text:
            return "REFUTES"
        elif "NOT ENOUGH INFO" in answer_text:
            return "NOT ENOUGH INFO"
    
    # If "Answer:" exists anywhere in the text, extract everything after it
    if "Answer:" in model_answer:
        after_answer = model_answer.split("Answer:", 1)[1].strip()  # Split only on first occurrence
        
        # Check if the extracted part has any of the expected labels
        if re.search(r'(SUPPORTS|REFUTES|NOT ENOUGH INFO)', after_answer, re.IGNORECASE):
            answer_text = after_answer.upper()
            if "SUPPORTS" in answer_text:
                return "SUPPORTS"
            elif "REFUTES" in answer_text:
                return "REFUTES"
            elif "NOT ENOUGH INFO" in answer_text:
                return "NOT ENOUGH INFO"
    
    # If no explicit Answer tag, try to find in the last few lines
    last_lines = model_answer.strip().split('\n')[-3:]  # Get last three lines
    for line in reversed(last_lines):
        if "SUPPORTS" in line.upper():
            return "SUPPORTS"
        elif "REFUTES" in line.upper():
            return "REFUTES"
        elif "NOT ENOUGH INFO" in line.upper():
            return "NOT ENOUGH INFO"
    
    # If all extraction methods fail, return None
    return None

def create_prompt_from_feverous(item: Dict[str, Any]) -> str:
    """Create prompt for FEVEROUS item"""
    # Get claim
    claim = item.get("claim", "")
    
    # Get evidence content
    evidence_content = item.get("evidence_content", {})
    
    # Format tables
    tables = evidence_content.get("tables", {})
    table_str = format_table(tables)
    
    # Format text
    sentences = evidence_content.get("sentences", {})
    text_str = format_sentences(sentences)
    
    # Create prompt
    prompt = COT_PROMPT_FEVEROUS_TEMPLATE.format(
        table=table_str,
        text=text_str,
        claim=claim
    )
    
    return prompt

def process_feverous_data(input_file, output_file, model_name, log_file, max_tokens=2048, temperature=0.7, start_from=0, api_port=8000):
    """Process FEVEROUS dataset"""
    logger = setup_logger(log_file)
    
    # Record start time
    start_time = time.time()
    logger.info(f"Started processing FEVEROUS data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
    
    # Read JSON file
    data_items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # For JSONL format, read line by line
            for line in f:
                if line.strip():  # Skip empty lines
                    item = json.loads(line.strip())
                    data_items.append(item)
        logger.info(f"Loaded {len(data_items)} data items from JSONL file")
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
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("id", f"item_{i}")
        claim = item.get("claim", "")
        expected_label = item.get("label", "")
        
        logger.info(f"Processing item {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        # Create prompt for FEVEROUS
        prompt = create_prompt_from_feverous(item)
        
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
            
            # Extract final answer
            extracted_answer = extract_answer_from_response(answer)
            if not extracted_answer:
                logger.warning(f"Unable to extract label from answer, using original response")
                extracted_answer = "NOT_EXTRACTED"
            
            # Check if prediction is correct
            is_correct = extracted_answer == expected_label
            if is_correct:
                correct_count += 1
            
            # Build result object
            result = {
                "id": item_id,
                "claim": claim,
                "prompt": prompt,
                "ground_truth": expected_label,
                "model_prediction": extracted_answer,
                "full_response": answer,
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
            logger.info(f"Claim: {claim}")
            logger.info(f"Expected label: {expected_label}")
            logger.info(f"Model prediction: {extracted_answer}")
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
                "claim": claim,
                "ground_truth": expected_label,
                "model_prediction": "Processing error",
                "is_correct": False,
                "error": str(e)
            }
            results.append(result)
        
        # Save intermediate results every 5 items or when error occurs
        if (i + 1) % 100 == 0 or error_count > 0:
            try:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(data_items)})")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
    
    # Calculate accuracy
    accuracy = correct_count / len(data_items) if len(data_items) > 0 else 0
    
    # Save results to JSON file
    try:
        # Sort results by ID before saving
        results.sort(key=lambda x: x.get("id", ""))
        
        # Add evaluation metrics
        final_results = {
            "results": results,
            "metadata": {
                "total_examples": len(data_items),
                "processed_examples": len(results),
                "correct_predictions": correct_count,
                "accuracy": accuracy,
                "model": model_name,
                "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")

    # Log summary information
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Processing failures: {error_count}/{len(data_items)}")
    logger.info(f"Correct predictions: {correct_count}/{len(data_items)}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
    logger.info("=" * 60)



def parse_arguments():
    parser = argparse.ArgumentParser(description='Process FEVEROUS dataset with LLM')
    
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
    input_file = os.path.join(base_path, "data/feverous/dev.jsonl")
    
    # 使用命令行参数，如果提供了参数则使用参数值，否则使用默认值
    output_file = args.output_file
    model_name = args.model_path
    log_file = args.log_file
    
    # 使用命令行参数的最大token数
    max_tokens = args.max_tokens
    
    # 使用命令行参数的起始索引
    start_from = args.start_from
    
    # 使用命令行参数的温度值
    temperature = args.temperature
    
    # 使用命令行参数的API端口
    api_port = args.api_port
    
    # 确保输出目录和日志目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 处理数据
    process_feverous_data(input_file, output_file, model_name, log_file, max_tokens, temperature, start_from, api_port)

if __name__ == "__main__":
    main()