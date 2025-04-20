import json
import os
import logging
import time
import random
import argparse
import sys
import re
from datetime import datetime
from openai import OpenAI
from llm import initialize_client, call_api_with_retry
from prompt import COT_PROMPT_TATQA_TEMPLATE

# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('tatqa_processor')
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

def create_prompt_from_tatqa(item, question_item):
    """Create prompt from TaTQA data item"""
    # Process table
    table_data = item["table"]["table"]
    table_str = ""
    for row in table_data:
        table_str += " | ".join([str(cell) for cell in row]) + "\n"
    
    # Process text paragraphs with paragraph numbers
    text_parts = []
    for i, para in enumerate(item["paragraphs"], 1):
        # Add paragraph number before each paragraph
        text_parts.append(f"Paragraph {i}: {para['text']}")
    text_str = "\n".join(text_parts)
    
    # Generate prompt
    prompt = COT_PROMPT_TATQA_TEMPLATE.format(
        table=table_str, 
        text=text_str, 
        question=question_item["question"]
    )
    
    return prompt


def extract_final_answer(response):

    if not response:
        return ""
    
    answer_tag_pattern = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    if answer_tag_pattern:
  
        answer_content = answer_tag_pattern.group(1).strip()
        
        if "Answer:" in answer_content:
            return answer_content.split("Answer:")[1].strip()
        return answer_content
    
    elif "Answer:" in response:
        return response.split("Answer:")[1].strip()
    
    return response

def find_gold_answer_with_info(test_data, question_text, table_uid):
    """Find gold answer and other info for a given question from the gold dataset"""
    for item in test_data:
        for question in item.get("questions", []):
            if question.get("question", "").strip() == question_text.strip():
                return {
                    "answer": question.get("answer", "No answer found"),
                    "answer_type": question.get("answer_type", "span"),
                    "answer_from": question.get("answer_from", "text"),
                    "scale": question.get("scale", ""),
                    "derivation": question.get("derivation", ""),
                    "rel_paragraphs": question.get("rel_paragraphs", []),
                    "req_comparison": question.get("req_comparison", False),
                    "facts": question.get("facts", []),
                    "consts": question.get("consts", []),
                    "mappings": question.get("mappings", [])
                }
    
    return {
        "answer": "No gold answer found",
        "answer_type": "",
        "answer_from": "",
        "scale": "",
        "derivation": "",
        "rel_paragraphs": [],
        "req_comparison": False,
        "facts": [],
        "consts": [],
        "mappings": []
    }

def process_tatqa_data(input_file, gold_file, output_file, model_name, log_file, max_tokens=2048, start_from=0, api_port=8000, temperature=0.7):
    """Process TaTQA dataset"""
    logger = setup_logger(log_file)
    
    # Record start time
    start_time = time.time()
    logger.info(f"Started processing TaTQA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Gold answer file: {gold_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
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
            test_data = json.load(f)
        logger.info(f"Loaded {len(test_data)} table data items from test file")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
    # Read gold answers JSON file
    gold_data = []
    try:
        with open(gold_file, 'r', encoding='utf-8') as f:
            gold_data = json.load(f)
        logger.info(f"Loaded gold answers with {len(gold_data)} items")
    except Exception as e:
        logger.error(f"Failed to read gold answer file: {e}")
        # Continue without gold answers
        logger.warning("Will proceed without gold answers")
    
    # Expand one-table-many-questions into multiple question items
    expanded_items = []
    question_count = 0
    
    for table_idx, table_item in enumerate(test_data):
        table_id = table_item["table"]["uid"]
        
        # Skip empty tables or those without questions
        if not table_item.get("questions"):
            continue
            
        for q_idx, question_item in enumerate(table_item["questions"]):
            question_count += 1
            # Create a more structured ID: table_index-question_index (e.g., T001-Q005)
            item_id = f"T{table_idx+1:03d}-Q{q_idx+1:03d}"
            
            expanded_items.append({
                "table_item": table_item,
                "question_item": question_item,
                "id": item_id,
                "original_ids": {
                    "table_uid": table_id,
                    "question_uid": question_item.get("uid", "")
                }
            })
    
    logger.info(f"Expanded into {len(expanded_items)} questions from {question_count} total questions")
    
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
    
    # Process each data item
    for i, item in enumerate(expanded_items[start_from:], start=start_from):
        item_id = item["id"]
        table_item = item["table_item"]
        question_item = item["question_item"]
        
        logger.info(f"Processing item {i+1}/{len(expanded_items)}... [ID: {item_id}]")
        
        # Find gold answer from gold data
        table_uid = item["original_ids"]["table_uid"]
        question_text = question_item["question"]
        # gold_answer = find_gold_answer(gold_data, question_text, table_uid)
        gold_answer_info = find_gold_answer_with_info(gold_data, question_text, table_uid)
        
        gold_answer = gold_answer_info["answer"]
        answer_type = gold_answer_info["answer_type"]
        answer_from = gold_answer_info["answer_from"]
        scale = gold_answer_info["scale"]
        derivation = gold_answer_info["derivation"]
        rel_paragraphs = gold_answer_info["rel_paragraphs"]
        req_comparison = gold_answer_info["req_comparison"]
        facts = gold_answer_info["facts"]
        consts = gold_answer_info["consts"]
        mappings = gold_answer_info["mappings"]
        
        # Create prompt for TaTQA
        prompt = create_prompt_from_tatqa(table_item, question_item)
        
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
            
            # Try to extract final answer if there is an "Answer:" marker
            final_answer = extract_final_answer(answer)
            # if "Answer:" in answer:
            #     final_answer = answer.split("Answer:")[1].strip()
            
            # Build result object
            result = {
                "id": item_id,
                "original_ids": {
                    "table_uid": table_uid,
                    "question_uid": question_item.get("uid", "")
                },
                "prompt": prompt,
                "question": question_item["question"],
                "model_answer": final_answer,
                "full_response": answer,
                "gold_answer": gold_answer,
                "answer_type": answer_type,
                "answer_from": answer_from,
                "scale": scale,
                "derivation": derivation,
                "rel_paragraphs": rel_paragraphs,
                "req_comparison": req_comparison,
                "facts": facts,
                "consts": consts,
                "mappings": mappings,
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
            logger.info(f"Question: {question_item['question']}")
            logger.info(f"Gold answer: {gold_answer}")
            logger.info(f"Answer type: {answer_type}")
            logger.info(f"Model answer: {final_answer}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing item {i+1}: {e}")
            # Record error information
            result = {
                "id": item_id,
                "original_ids": {
                    "table_uid": table_uid,
                    "question_uid": question_item.get("uid", "")
                },
                "question": question_item["question"],
                "gold_answer": gold_answer,
                "answer_type": answer_type,
                "answer_from": answer_from,
                "scale": scale,
                "derivation": derivation,
                "rel_paragraphs": rel_paragraphs,
                "req_comparison": req_comparison,
                "facts": facts,
                "consts": consts,
                "mappings": mappings,
                "model_answer": f"Processing error: {str(e)}",
                "error": str(e)
            }
            results.append(result)
        
        # Save intermediate results every 5 items or when error occurs
        if (i + 1) % 5 == 0 or error_count > 0:
            try:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(expanded_items)})")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
    
    # Save results to JSON file
    try:
        # Sort results by ID before saving
        results.sort(key=lambda x: x["id"])
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")

    # Log summary information
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(expanded_items)}")
    logger.info(f"Processing failures: {error_count}/{len(expanded_items)}")
    if len(expanded_items) > 0:
        logger.info(f"Success rate: {success_count/len(expanded_items)*100:.2f}%")
    logger.info("=" * 60)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Process TaTQA dataset with LLM')
    
    parser.add_argument('--api_port', type=int, default=8000, help='API port for local model server')
    parser.add_argument('--output_file', type=str, help='Path to save results')
    parser.add_argument('--model_path', type=str, help='Model path or identifier')
    parser.add_argument('--data_path', type=str, help='Path to input data file')
    parser.add_argument('--gold_file', type=str, help='Path to gold answer file')
    parser.add_argument('--log_file', type=str, help='Path to log file')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for model generation')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum tokens for model output')
    parser.add_argument('--start_from', type=int, default=0, help='Start processing from this index')
    parser.add_argument('--base_path', type=str, help='Base path for the project')
    
    return parser.parse_args()

def main():
  
    args = parse_arguments()
    
    if args.base_path and os.path.exists(args.base_path):
        base_path = args.base_path
    
    if not base_path:
        print("Error: Unable to find project root directory")
        exit(1)
    
    print(f"Using root path: {base_path}")
    
    input_file = os.path.join(base_path, "data/tatqa/tatqa_dataset_test.json")
    gold_file = os.path.join(base_path, "data/tatqa/tatqa_dataset_test_gold.json")
    output_file = args.output_file

    model_name = args.model_path
    
    log_file = args.log_file 
    
    max_tokens = args.max_tokens
    temperature = args.temperature
    
    start_from = args.start_from
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
 
    process_tatqa_data(input_file, gold_file, output_file, model_name, log_file, max_tokens, start_from, api_port=args.api_port, temperature=temperature)

if __name__ == "__main__":
    main()