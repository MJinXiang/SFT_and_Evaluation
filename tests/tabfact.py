import json
import os
import logging
import time
import re
import sys
from datetime import datetime
# sys.path.append("/mnt/usercache/mengjinxiang/Project/LLaMA-Factory-main/baseline")
from llm import initialize_client, call_api_with_retry
from prompt import COT_PROMPT_TABFACT_TEMPLATE


# Setup logging
def setup_logger(log_file):
    """Set up logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('tabfact_processor')
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

def create_prompt_from_tabfact(item):
    """Create prompt for TabFact data"""
    table_str = ""
    if "table" in item:
        # Create string representation of the table
        for row in item["table"]:
            table_str += " | ".join([str(cell) for cell in row]) + "\n"
    
    # Use the TabFact template
    prompt = COT_PROMPT_TABFACT_TEMPLATE.format(table=table_str, claim=item["question"])
    
    return prompt

def extract_answer_from_response(response):
    """Extract final answer (SUPPORTS or REFUTES) from response"""
    # Use regex to find "Answer: xxx" pattern
    match = re.search(r'Answer:\s*(SUPPORTS|REFUTES)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()  # Return uppercase answer
    return None

def process_tabfact_data(input_file, output_file, model_name, log_file, max_tokens=2048, start_from=0):
    """Process TabFact dataset"""
    logger = setup_logger(log_file)
    
    # Record start time
    start_time = time.time()
    logger.info(f"Start processing TabFact data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Starting from index {start_from}")
    
    # Initialize model client
    try:
        client_info = initialize_client({"model_path": model_name})
        logger.info(f"Model client initialized successfully, type: {client_info['model_type']}")
    except Exception as e:
        logger.error(f"Model client initialization failed: {e}")
        return
    
    # Read JSON file
    data_items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_items = json.load(f)
        logger.info(f"Loaded {len(data_items)} data items")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
    # Check if intermediate results exist, load if available
    results = []
    if start_from > 0 and os.path.exists(f"{output_file}.temp"):
        try:
            with open(f"{output_file}.temp", 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded intermediate results with {len(results)} records")
        except Exception as e:
            logger.error(f"Failed to load intermediate results: {e}, starting from scratch")
            start_from = 0
    
    success_count = len(results)
    error_count = 0
    
    # Process each data item
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("id", f"item-{i}")
        logger.info(f"Processing item {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        # Create prompt for TabFact
        prompt = create_prompt_from_tabfact(item)
        
        # Set user message
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Record API call start time
            call_start = time.time()
            
            # Call API
            api_result = call_api_with_retry(
                client_info=client_info,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=1.0,
                max_retries=10
            )
            
            # Initialize thinking variable
            thinking = None
            
            # Process API result - check if it's a deepseek-r1 model
            if client_info["model_type"] in ["deepseek-r1", "deepseek-r1-inner"]:
                # deepseek-r1 returns three values: success flag, answer content, and reasoning content
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
            
            # Extract final answer (SUPPORTS or REFUTES)
            extracted_answer = extract_answer_from_response(answer)
            if not extracted_answer:
                logger.warning(f"Unable to extract label from answer, using original response")
                extracted_answer = answer
            
            # Convert answer to boolean or number to match true answer format
            # TabFact typically uses 1 for SUPPORTS, 0 for REFUTES
            model_label = 1 if extracted_answer == "SUPPORTS" else 0
            
            # Build result object
            result = {
                "id": item_id,
                "table_id": item.get("table_id", ""),
                "table_title": item.get("table_title", ""),
                "prompt": prompt,
                "claim": item["question"],
                "ground_truth": item["answer"],  # Original label (0 or 1)
                "extracted_answer": extracted_answer,  # Extracted text answer (SUPPORTS or REFUTES)
                "model_label": model_label,  # Label converted to number
                "full_response": answer,  # Complete model response
                "processing_time": call_time,
                "token_usage": token_info,
                "is_correct": model_label == item["answer"]  # Whether it matches the true label
            }
            
            # Add thinking content field (for deepseek-r1 model)
            if thinking is not None:
                result["thinking"] = thinking
                logger.info(f"Model thinking process: {thinking}")
            
            results.append(result)
            success_count += 1
            
            # Log detailed information
            logger.info(f"Claim: {item['question']}")
            logger.info(f"Ground truth: {item['answer']} ({'SUPPORTS' if item['answer'] == 1 else 'REFUTES'})")
            logger.info(f"Model judgment: {model_label} ({extracted_answer})")
            logger.info(f"Judgment correct: {model_label == item['answer']}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing item {i+1}: {e}")
            # Record error information
            result = {
                "id": item_id,
                "table_id": item.get("table_id", ""),
                "table_title": item.get("table_title", ""),
                "claim": item["question"],
                "ground_truth": item["answer"],
                "model_label": "Processing error",
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
    
    # Calculate overall accuracy
    correct_count = sum(1 for result in results if result.get("is_correct", False))
    accuracy = correct_count / len(results) if results else 0
    
    # Save results to JSON file
    try:
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
    logger.info(f"Processing complete! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Processing failed: {error_count}/{len(data_items)}")
    logger.info(f"Accuracy: {accuracy*100:.2f}%")
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
        print("Error: Cannot find project root directory")
        exit(1)
    
    print(f"Using root path: {base_path}")
    
    # Parameter settings
    input_file = os.path.join(base_path, "data/tabfact/test.json")
    output_file = os.path.join(base_path, "results/tabfact/tabfact_sft_ppo_results.json")
    model_name = "/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"  # Can use "deepseek-r1" or local model path
    log_file = os.path.join(base_path, "results/tabfact/logs/tabfact_sft_ppo_processing.log")
    max_tokens = 2048  # Maximum output tokens
    
    # Checkpoint resumption parameter - which data item to start from (0 means start from beginning)
    start_from = 0
    
    # Ensure output and log directories exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Process data
    process_tabfact_data(input_file, output_file, model_name, log_file, max_tokens, start_from)

if __name__ == "__main__":
    main()