import json
import os
import logging
import time
import re
import sys
from datetime import datetime
from llm import initialize_client, call_api_with_retry
from prompt import COT_PROMPT_MMQA_TEMPLATE


def setup_logger(log_file):
    """Set up logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('mmqa_processor')
    logger.setLevel(logging.INFO)
    
    # Create file handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def format_table(table):
    """Format a table from MMQA format to a readable string"""
    result = []
    
    # Add table title if available
    if "title" in table:
        result.append(f"Title: {table['title']}")
    
    # Format columns and data into a markdown-style table
    if "columns" in table and "data" in table:
        # Add header
        result.append(" | ".join(table["columns"]))
        # result.append("-" * (sum(len(col) for col in table["columns"]) + 3 * (len(table["columns"]) - 1)))
        
        # Add rows
        for row in table["data"]:
            result.append(" | ".join(str(cell) for cell in row))
    
    return "\n".join(result)

def create_prompt_from_mmqa(item):
    """Create a prompt for MMQA data"""
    # Get question
    question = item.get("question", "")
    
    # Get tables
    tables = item.get("tables", {})
    table1 = format_table(tables.get("table1", {}))
    table2 = format_table(tables.get("table2", {}))
    
    # Generate prompt
    prompt = COT_PROMPT_MMQA_TEMPLATE.format(
        question=question,
        table1=table1,
        table2=table2
    )
    
    return prompt

def extract_answer_from_response(model_answer):
    """
    Extract the final answer from the model response
    """
    # If response is empty, return empty string
    if not model_answer:
        return ""
    
    # Check if it contains "Answer:" marker
    answer_pattern = re.search(r'Answer:\s*(.*?)(?:$|\.|\n)', model_answer, re.DOTALL)
    if answer_pattern:
        return answer_pattern.group(1).strip()
    
    # If no standard format is found, return the last line of content
    lines = model_answer.strip().split('\n')
    return lines[-1].strip()

def process_mmqa_data(input_file, output_file, model_name, log_file, max_tokens=2048, temperature=0.0, start_from=0, max_prompt_length=40000):
    """Process MMQA dataset"""
    logger = setup_logger(log_file)
    
    # Record start time
    start_time = time.time()
    logger.info(f"Started processing MMQA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Starting from index: {start_from}")
    logger.info(f"Maximum prompt length: {max_prompt_length} characters")
    
    # Initialize model client
    try:
        client_info = initialize_client({"model_path": model_name})
        model_type = client_info["model_type"]
        logger.info(f"Model client initialized successfully, type: {model_type}")
    except Exception as e:
        logger.error(f"Model client initialization failed: {e}")
        return
    
    # Read JSON file
    data_items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_items = json.load(f)
        logger.info(f"Loaded {len(data_items)} questions from input file")
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
    
    # Prepare evaluation result format
    predictions = {}
    
    # Process each data item
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("uid", f"item-{i}")
        
        logger.info(f"Processing item {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        # Get question and ground truth
        question = item.get("question", "")
        ground_truth = item.get("answer", {})
        
        # Extract ground truth answer for logging
        if isinstance(ground_truth, dict) and "data" in ground_truth:
            ground_truth_str = str(ground_truth["data"])
        else:
            ground_truth_str = str(ground_truth)
        
        # Create prompt
        # prompt = create_prompt_from_mmqa(item)

        try:
            prompt = create_prompt_from_mmqa(item)
            
            # 检查prompt长度并在必要时截断
            if len(prompt) > max_prompt_length:
                logger.warning(f"Prompt for item {item_id} exceeds maximum length ({len(prompt)} > {max_prompt_length}). Skipping this example...")
                # 记录跳过的样本信息
                result = {
                    "uid": item_id,
                    "question": question,
                    "ground_truth": ground_truth,
                    "model_answer": "Skipped due to excessive prompt length",
                    "extracted_answer": "",
                    "processing_time": 0,
                    "prompt_length": len(prompt),
                    "skipped": True,
                    "reason": f"Prompt length ({len(prompt)}) exceeds maximum allowed ({max_prompt_length})"
                }
                
                results.append(result)
                predictions[item_id] = ""  # 添加空预测以避免评估错误
                
                # 继续下一个样本
                continue
                
        except Exception as e:
            logger.error(f"Error creating prompt for item {item_id}: {str(e)}")
            error_count += 1
            # 记录错误信息
            result = {
                "uid": item_id,
                "question": question,
                "model_answer": f"Error creating prompt: {str(e)}",
                "extracted_answer": "",
                "error": str(e)
            }
            
            results.append(result)
            predictions[item_id] = ""
            continue
        
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
                top_p=1.0,
                max_retries=10
            )
            
            # Initialize thinking variable to avoid undefined error
            thinking = None
            
            # Process API result - check if it's a deepseek-r1 model
            if model_type in ["deepseek-r1", "deepseek-r1-inner"]:
                # deepseek-r1 returns three values: success flag, answer content, and reasoning content
                success, answer, thinking = api_result
            else:
                # Other models return two values: success flag and answer content
                success, answer = api_result

            if not success:
                error_msg = f"API call failed: {answer}"
                if "provider_failed" in str(answer):
                    error_msg += " - Provider service error, possibly server overload or temporary outage"
                elif "context_length_exceeded" in str(answer):
                    error_msg += " - Input context length exceeded model's maximum limit"
                elif "rate_limit_exceeded" in str(answer):
                    error_msg += " - API rate limit reached, consider reducing request frequency"
                
                logger.error(error_msg)
                raise Exception(error_msg)
            
            # Calculate API call time
            call_time = time.time() - call_start
            
            # Process returned response - extract token usage
            token_info = {}
            if model_type == "openai" and hasattr(answer, 'usage'):
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
            
            # Build result object
            result = {
                "uid": item_id,
                "question": question,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "model_answer": answer,
                "extracted_answer": extracted_answer,
                "processing_time": call_time,
                "token_usage": token_info
            }
            
            # Add thinking content field (applicable for deepseek-r1 model)
            if thinking is not None:
                result["reasoning"] = thinking
            
            results.append(result)
            success_count += 1
            
            # Add to predictions dictionary for evaluation format
            predictions[item_id] = extracted_answer
            
            # Log detailed information
            logger.info(f"Question: {question}")
            logger.info(f"Ground truth: {ground_truth_str}")
            logger.info(f"Model answer: {extracted_answer}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            import traceback
            stack_trace = traceback.format_exc()
            logger.error(f"Error processing item {i+1}: {e}")
            logger.error(f"Detailed error trace:\n{stack_trace}")
            # logger.error(f"Error processing item {i+1}: {e}")
            # Record error information
            result = {
                "uid": item_id,
                "question": question,
                "model_answer": f"Processing error: {str(e)}",
                # "error": str(e)
                "error_trace": stack_trace.split("\n")[-5:]  # 只保存最后几行跟踪信息
            }
            results.append(result)
            
            # Add empty prediction to avoid evaluation error
            predictions[item_id] = ""
        
        # Save intermediate results every 5 items or when an error occurs
        if (i + 1) % 5 == 0 or error_count > 0:
            try:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(data_items)})")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
    
    # Save detailed results to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save detailed results file: {e}")
    
    # Save evaluation format output (only contains question ID and predicted answer)
    eval_output_file = output_file.replace('.json', '_eval.json')
    try:
        # Create data structure for evaluation format
        eval_data = [
            {
                "uid": uid,
                "pred": pred
            }
            for uid, pred in predictions.items()
        ]
        
        with open(eval_output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Evaluation format results saved to {eval_output_file}")
    except Exception as e:
        logger.error(f"Failed to save evaluation format results: {e}")

    # Log summary information
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Processing failures: {error_count}/{len(data_items)}")
    logger.info("=" * 60)

def main():
    # Detect correct root path
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
    
    # Parameter settings
    input_file = os.path.join(base_path, "data/MMQA/two_table_with_keys_with_sql/test_fixed.json")  # MMQA test set
    output_file = os.path.join(base_path, "results/mmqa/mmqa2qa_sft_results.json")
    model_name = "/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"  # Can use "deepseek-r1" or local model path  /mnt/usercache/huggingface/Qwen2.5-3B-Instruct
    log_file = os.path.join(base_path, "results/mmqa/logs/mmqa2qa_sft_processing.log")
    max_tokens = 4096  # Maximum output token count
    temperature = 0.6  # Set to 0 for deterministic output
    max_prompt_length = 40000 
    
    # Checkpoint parameter - which data item to start from (0 means start from beginning)
    start_from = 0
    
    # Ensure output directory and log directory exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Process data
    process_mmqa_data(input_file, output_file, model_name, log_file, max_tokens, temperature, start_from, max_prompt_length)

if __name__ == "__main__":
    main()