import json
import os
import logging
import time
import re
import argparse
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
from llm import initialize_client, call_api_with_retry
from prompt import COT_PROMPT_TOTTO_TEMPLATE


# Setup logging
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('totto_processor')
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

def format_table_with_highlights(table_data, highlighted_cells):
    """Format table data as a string, marking highlighted cells"""
    if not table_data:
        return ""
    
    result = ""
    for row_idx, row in enumerate(table_data):
        row_str = []
        for col_idx, cell in enumerate(row):
            # Check if this is a highlighted cell
            is_highlighted = [row_idx, col_idx] in highlighted_cells
            if is_highlighted:
                row_str.append(f"<hl>{cell}</hl>")
            else:
                row_str.append(str(cell))
        
        result += " | ".join(row_str) + "\n"
    
    return result.strip()

def create_prompt_from_totto(item: Dict[str, Any]) -> str:
    """Create prompt for ToTTo item"""
    # Get table, context, and highlighted cells
    table_data = item.get("table", [])
    highlighted_cells = item.get("highlighted_cells", [])
    page_title = item.get("table_page_title", "")
    section_title = item.get("table_section_title", "")
    section_text = item.get("table_section_text", "")
    
    # Format table with highlighted cells
    table_str = format_table_with_highlights(table_data, highlighted_cells)
    
    # Create prompt
    prompt = COT_PROMPT_TOTTO_TEMPLATE.format(
        table=table_str,
        page_title=page_title,
        section_title=section_title,
        section_text=section_text
    )
    
    return prompt



def extract_answer_from_response(model_answer):
    """Extract the final answer from the model response, supporting multiple formats"""
    if not model_answer:
        return ""
    
    # Try to extract answer from <answer> tags
    answer_tag_pattern = re.search(r'<answer>(.*?)</answer>', model_answer, re.DOTALL)
    if answer_tag_pattern:
        answer_content = answer_tag_pattern.group(1).strip()
        
        # Check if there's an "Answer:" prefix inside the tags
        if "Answer:" in answer_content:
            return answer_content.split("Answer:", 1)[1].strip()
        return answer_content
    
    # Try to match "Answer: xxx" pattern, capturing the entire sentence
    # This pattern specifically looks for "Answer:" and captures everything after it
    answer_pattern = re.search(r'Answer:\s*(.*?)(?:\Z)', model_answer, re.IGNORECASE | re.DOTALL)
    if answer_pattern:
        complete_answer = answer_pattern.group(1).strip()
        # Remove any trailing code blocks or irrelevant content
        if "```" in complete_answer:
            complete_answer = complete_answer.split("```")[0].strip()
        return complete_answer
    
    # If no explicit marker, try to use the last paragraph as the answer
    paragraphs = model_answer.strip().split('\n\n')
    for paragraph in reversed(paragraphs):
        cleaned_paragraph = paragraph.strip()
        if cleaned_paragraph and not cleaned_paragraph.startswith('```') and not cleaned_paragraph.endswith('```'):
            return cleaned_paragraph
    
    # If all extraction methods fail, return the original response
    return model_answer

def process_totto_data(input_file, output_file, model_name, log_file, max_tokens=2048, temperature=0.0, start_from=0, api_port=8000):
    """Process ToTTo dataset"""
    logger = setup_logger(log_file)
    
    # Record start time
    start_time = time.time()
    logger.info(f"Started processing ToTTo data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"API port: {api_port}")
    logger.info(f"Starting from index: {start_from}")
    
    # Initialize model client
    try:
        client_info = initialize_client({"model_path": model_name, "api_port": api_port})
        model_type = client_info["model_type"]
        logger.info(f"Model client initialized successfully, type: {model_type}")
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
        logger.info(f"Loaded {len(data_items)} examples from JSONL file")
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
    
    # Process each data item
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("example_id", f"item-{i}")
        
        logger.info(f"Processing example {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        # Get reference sentences if available
        reference_sentences = []
        if "sentence_annotations" in item:
            reference_sentences = [
                annotation.get("final_sentence", "") 
                for annotation in item.get("sentence_annotations", [])
                if "final_sentence" in annotation
            ]
        
        # Create prompt for ToTTo
        prompt = create_prompt_from_totto(item)
        
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
            
            # Process API result - check if it's deepseek-r1 model
            if model_type in ["deepseek-r1", "deepseek-r1-inner"]:
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
            generated_sentence = extract_answer_from_response(answer)
            
            # Build result object
            result = {
                "example_id": item_id,
                "prompt": prompt,
                "reference_sentences": reference_sentences,
                "model_full_response": answer,
                "generated_sentence": generated_sentence,
                "processing_time": call_time,
                "token_usage": token_info
            }
            
            # Add thinking content field (for deepseek-r1 model)
            if thinking is not None:
                result["reasoning"] = thinking
            
            results.append(result)
            success_count += 1
            
            # Log detailed information
            logger.info(f"Page title: {item.get('table_page_title', '')}")
            logger.info(f"Reference sentences: {reference_sentences}")
            logger.info(f"Generated sentence: {generated_sentence}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing item {i+1}: {e}")
            # Record error information
            result = {
                "example_id": item_id,
                "reference_sentences": reference_sentences,
                "generated_sentence": f"Processing error: {str(e)}",
                "error": str(e)
            }
            results.append(result)
        
        # Save intermediate results every 5 items or when error occurs
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
    

    # Log summary information
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Processing failures: {error_count}/{len(data_items)}")
    logger.info("=" * 60)



def parse_arguments():
    parser = argparse.ArgumentParser(description='Process ToTTo dataset with LLM')
    
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
    input_file = os.path.join(base_path, "data/totto/dev.jsonl")  # ToTTo test set
    
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
    process_totto_data(input_file, output_file, model_name, log_file, max_tokens, temperature, start_from, api_port)

if __name__ == "__main__":
    main()