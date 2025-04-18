import json
import os
import logging
import time
import random
import sys
from datetime import datetime
from openai import OpenAI

# Setup logging
def setup_logger(log_file):
    """Set up a logger with file and console handlers"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('table_analysis')
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

def call_api_with_retry(client, messages, model_path, temperature, max_tokens=4096, max_retries=10, logger=None):
    """Call API with custom retry mechanism"""
    attempt = 0
    last_exception = None
    
    while attempt < max_retries:
        try:
            attempt += 1
            if logger and attempt > 1:
                logger.info(f"Attempt {attempt}/{max_retries} to call API...")
            
            return client.chat.completions.create(
                messages=messages,
                model=model_path,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
        except Exception as e:
            last_exception = e
            if logger:
                logger.warning(f"API call failed (attempt {attempt}/{max_retries}): {e}")
            
            wait_time = min(2 ** attempt + random.uniform(0, 1), 60)
            
            if logger:
                logger.info(f"Waiting {wait_time:.2f} seconds before retrying...")
            
            time.sleep(wait_time)
    
    if logger:
        logger.error(f"Reached maximum retry attempts ({max_retries}), giving up")
    
    raise last_exception

def process_table_data(input_file, output_file, model_path, log_file, max_tokens=4096, start_from=0):
    """Process table dataset, use instructions as prompts to send to LLM, and save results"""
    # Set up logging
    logger = setup_logger(log_file)
    
    # Record start time
    start_time = time.time()
    logger.info(f"Starting table data processing: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_path}")
    logger.info(f"Max output tokens: {max_tokens}")
    logger.info(f"Starting from index: {start_from}")
    
    # Initialize OpenAI client
    try:
        client = OpenAI(api_key="0", base_url="http://0.0.0.0:8000/v1")
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"OpenAI client initialization failed: {e}")
        return
    
    # Read JSONL file
    data_items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data_items.append(json.loads(line))
        logger.info(f"Loaded {len(data_items)} data items")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
    # Check if intermediate results exist and load them
    results = []
    if start_from > 0 and os.path.exists(f"{output_file}.temp"):
        try:
            with open(f"{output_file}.temp", 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded intermediate results containing {len(results)} records")
        except Exception as e:
            logger.error(f"Failed to load intermediate results: {e}, starting from scratch")
            start_from = 0
    
    success_count = len(results)
    error_count = 0
    
    # Process each data item
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("id", f"item-{i}")
        logger.info(f"Processing item {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        # Use instruction field as prompt
        if "instruction" not in item:
            logger.warning(f"Data item {i+1} missing instruction field, skipping")
            continue
        
        # Get instruction as prompt
        prompt = item["instruction"]
        
        # Create user message
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Record API call start time
            call_start = time.time()
            
            # Use custom retry mechanism to call API
            response = call_api_with_retry(
                client=client, 
                messages=messages, 
                model_path=model_path, 
                temperature=0.2,
                max_tokens=max_tokens,
                max_retries=10,
                logger=logger
            )
            
            # Calculate API call time
            call_time = time.time() - call_start
            
            # Extract answer
            answer = response.choices[0].message.content

            # Record token usage
            token_info = {}
            if hasattr(response, 'usage'):
                token_info = {
                    "completion_tokens": getattr(response.usage, 'completion_tokens', 'N/A'),
                    "prompt_tokens": getattr(response.usage, 'prompt_tokens', 'N/A'),
                    "total_tokens": getattr(response.usage, 'total_tokens', 'N/A')
                }
            
            # Build result object
            result = {
                "id": item_id,
                "qtype": item.get("qtype", ""),
                "qsubtype": item.get("qsubtype", ""),
                "instruction": item["instruction"],
                "answer": item.get("answer", ""),
                "model_answer": answer,
                "processing_time": call_time,
                "token_usage": token_info
            }
            
            results.append(result)
            success_count += 1
            
            # Log detailed information
            logger.info(f"Question type: {item.get('qtype', '')}/{item.get('qsubtype', '')}")
            logger.info(f"Expected answer: {item.get('answer', '')}")
            logger.info(f"Model answer: {answer[:150]}..." if len(answer) > 150 else f"Model answer: {answer}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing data item {i+1}: {e}")
            # Record error information
            result = {
                "id": item_id,
                "qtype": item.get("qtype", ""),
                "qsubtype": item.get("qsubtype", ""),
                "instruction": item["instruction"],
                "expected_answer": item.get("answer", ""),
                "model_answer": f"Processing error: {str(e)}",
                "error": str(e)
            }
            results.append(result)
        
        # Save intermediate results every 5 items or when an error occurs
        if (i + 1) % 5 == 0 or error_count > 0:
            try:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(data_items)})")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
    
    # Save results to JSON file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")
    
    # Log summary information
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing complete! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Processing failures: {error_count}/{len(data_items)}")
    if len(data_items) > 0:
        logger.info(f"Success rate: {success_count/len(data_items)*100:.2f}%")
    logger.info("=" * 60)

def main():
    # Detect correct root path
    base_paths = [
        "/mnt/usercache/mengjinxiang/Project/LLaMA-Factory-main",
        # "/netcache/mengjinxiang/Project/LLaMA-Factory-main"
    ]
    
    base_path = None
    for path in base_paths:
        if os.path.exists(path):
            base_path = path
            break
    
    if not base_path:
        print("Error: Unable to find project root directory")
        exit(1)
    
    print(f"Using root path: {base_path}")
    
    # Parameter settings
    input_file = os.path.join(base_path, "data/tablebench/test.jsonl")
    output_file = os.path.join(base_path, "results/tablebench_sft_results.json")
    model_name = "/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"  # Model identifier used in API
    log_file = os.path.join(base_path, "results/tablebench_sft_processing.log")
    max_tokens = 4096  # Maximum output tokens
    
    # Checkpoint parameter - from which data item to start processing (0 means start from beginning)
    start_from = 0
    
    # Ensure output directory and log directory exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Process data
    process_table_data(input_file, output_file, model_name, log_file, max_tokens, start_from)

if __name__ == "__main__":
    main()