
import json
import os
import logging
import time
import random
import argparse
from datetime import datetime
from openai import OpenAI
import pandas as pd
from prompt import COT_PROMPT_WIKISQL_TEMPLATE
from llm import initialize_client, call_api_with_retry # type: ignore


def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('wikisql_processor')
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

def format_table_for_prompt(table_data):
    header = table_data["header"]
    rows = table_data["rows"]
    
    df = pd.DataFrame(rows, columns=header)
    
    table_str = df.to_string(index=False)
    
    return table_str

def process_wikisql(input_file, output_file, model_name, log_file, max_tokens=2048, start_from=0, api_port=8000, temperature=0.2):

    logger = setup_logger(log_file)
    
    start_time = time.time()
    logger.info(f"Starting WikiSQL data processing: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"API port: {api_port}")
    logger.info(f"Starting from index: {start_from}")
    
    
    try:
        client_info = initialize_client({"model_path": model_name, "api_port": api_port})
        logger.info("Model client initialized successfully")
    except Exception as e:
        logger.error(f"Model client initialization failed: {e}")
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
            logger.error(f"Failed to load intermediate results: {e}, starting from beginning")
            start_from = 0
    
    success_count = len(results)
    error_count = 0
    
    # Process each data item
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("id", f"item-{i}")
        logger.info(f"Processing item {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        # Format table data
        formatted_table = format_table_for_prompt(item["table"])
        
        # Build prompt
        prompt = COT_PROMPT_WIKISQL_TEMPLATE.format(
            table=formatted_table,
            question=item["question"]
        )
        
        # Create user message
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # Record API call start time
            call_start = time.time()
            
            api_result = call_api_with_retry(
                client_info=client_info,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                max_retries=10
            )

          
            thinking = None
         
            if client_info["model_type"] == "deepseek-r1" or client_info["model_type"] == "deepseek-r1-inner":
                success, answer, thinking = api_result
            else:
                success, answer = api_result


            if not success:
                raise Exception(f"API调用失败: {answer}")
                
            
            # Calculate API call time
            call_time = time.time() - call_start
            
         
            if client_info["model_type"] == "openai":
                # Extract token usage if available
                token_info = {}
                if hasattr(answer, 'usage'):
                    token_info = {
                        "completion_tokens": getattr(answer.usage, 'completion_tokens', 'N/A'),
                        "prompt_tokens": getattr(answer.usage, 'prompt_tokens', 'N/A'),
                        "total_tokens": getattr(answer.usage, 'total_tokens', 'N/A')
                    }
            
                answer = answer.choices[0].message.content
            else:
                token_info = {"note": "Token usage not available for this model type"}
            
            
            # Prepare expected SQL query
            expected_sql = item["sql"]["human_readable"] if "sql" in item else "N/A"
            
          
            result = {
                "id": item_id,
                "question": item["question"],
                "truth_sql": expected_sql,
                "truth_answer": item["sql"],
                "model_answer": answer,
                "processing_time": call_time,
                "token_usage": token_info
            }

        
            if thinking is not None:
                result["think"] = thinking
                logger.info(f"Model thinking: {thinking}")
    
            
            results.append(result)
            success_count += 1
            
            # Log detailed information
            logger.info(f"Question: {item['question']}")
            logger.info(f"Expected SQL: {expected_sql}")
            logger.info(f"Model answer: {answer}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing item {i+1}: {e}")
            # Record error information
            result = {
                "id": item_id,
                "question": item["question"],
                "truth_sql": expected_sql if "sql" in item else "N/A",
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

    # Evaluate SQL generation quality
    evaluate_sql_generation(results, logger)

def evaluate_sql_generation(results, logger):
    """Evaluate SQL generation quality"""
    total = len(results)
    if total == 0:
        logger.info("No results to evaluate")
        return
    
    # Calculate how many results contain valid SQL queries
    sql_count = 0
    exact_match_count = 0
    
    for result in results:
        model_answer = result.get('model_answer', '')
        expected_sql = result.get('truth_sql', '')
        
        if '```sql' in model_answer:
            sql_count += 1
            
            # Extract SQL statement from the answer
            try:
                # Extract content between ```sql and ```
                sql_start = model_answer.find('```sql') + 6
                sql_end = model_answer.find('```', sql_start)
                if sql_end != -1:
                    extracted_sql = model_answer[sql_start:sql_end].strip()
                    
                    # Normalize SQL queries (simplified approach, might need more complex SQL parsing)
                    normalized_extracted = extracted_sql.lower().replace(' ', '').replace('\n', '')
                    normalized_expected = expected_sql.lower().replace(' ', '').replace('\n', '')
                    
                    # Check for exact match
                    if normalized_extracted == normalized_expected:
                        exact_match_count += 1
            except:
                pass
    
    # Calculate metrics
    sql_inclusion_rate = sql_count / total * 100 if total > 0 else 0
    exact_match_rate = exact_match_count / total * 100 if total > 0 else 0
    
    # Log evaluation results
    logger.info("SQL generation quality assessment:")
    logger.info(f"Total samples: {total}")
    logger.info(f"Samples with SQL code blocks: {sql_count} ({sql_inclusion_rate:.2f}%)")
    logger.info(f"Exact SQL matches: {exact_match_count} ({exact_match_rate:.2f}%)")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process WikiSQL dataset with LLM')
    
    parser.add_argument('--api_port', type=int, default=8000, help='API port for local model server')
    parser.add_argument('--output_file', type=str, help='Path to save results')
    parser.add_argument('--model_path', type=str, help='Model path or identifier')
    parser.add_argument('--log_file', type=str, help='Path to log file')
    parser.add_argument('--temperature', type=float, default=0.6, help='Temperature for model generation')
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
    input_file = os.path.join(base_path, "data/wikisql/wikisql_test.json")
    
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
    process_wikisql(input_file, output_file, model_name, log_file, max_tokens, start_from, api_port, temperature)

if __name__ == "__main__":
    main()