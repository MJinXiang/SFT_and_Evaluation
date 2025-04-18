import json
import os
import logging
import time
import re
import sys
from datetime import datetime
from llm import initialize_client, call_api_with_retry
from prompt import COT_PROMPT_MULTHIERTT_TEMPLATE


def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('multhiertt_processor')
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def format_tables_for_prompt(tables_data):

    result = []
    
    table_id=0
    for table_name, table in tables_data.items():
        result.append(f"Table {table_id}:")
        table_id+=1
        
  
        formatted_table = []
        for row in table:
            formatted_table.append(" | ".join([str(cell) for cell in row]))
        
        result.append("\n".join(formatted_table))
        result.append("")  
    
    return "\n".join(result)

def format_table_description(table_desc):
   
    result = []
    
    # 对每个表格描述项进行格式化
    for key, desc in table_desc.items():
        result.append(f"{desc}")
    
    return "\n".join(result)

def format_paragraphs(paragraphs):
   
    if not paragraphs:
        return "No additional text provided."
    
    return "\n\n".join([f"Paragraph {i+1}: {para}" for i, para in enumerate(paragraphs)])

def create_prompt_from_multhiertt(item):
 
    question = item.get("question", "")

    tables_str = format_tables_for_prompt(item.get("tables", {}))
    
    table_desc_str = format_table_description(item.get("table_description", {}))
    
    text_str = format_paragraphs(item.get("paragraphs", []))
    
    prompt = COT_PROMPT_MULTHIERTT_TEMPLATE.format(
        question=question,
        table=tables_str,
        table_description=table_desc_str,
        text=text_str
    )
    
    return prompt

def extract_answer_from_response(model_answer):

    if not model_answer:
        return ""
    
    answer_pattern = re.search(r'Answer:\s*(.*?)(?:$|\.|\n)', model_answer, re.DOTALL)
    if answer_pattern:
        return answer_pattern.group(1).strip()
    
    lines = model_answer.strip().split('\n')
    return lines[-1].strip()

def process_multhiertt_data(input_file, output_file, model_name, log_file, max_tokens=2048, temperature=0.0, start_from=0):
    logger = setup_logger(log_file)
  
    start_time = time.time()
    logger.info(f"Started processing MultHier-TT data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"Temperature: {temperature}")
    logger.info(f"Starting from index: {start_from}")
    
    try:
        client_info = initialize_client({"model_path": model_name})
        model_type = client_info["model_type"]
        logger.info(f"Model client initialized successfully, type: {model_type}")
    except Exception as e:
        logger.error(f"Model client initialization failed: {e}")
        return
    
    data_items = []
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_items = json.load(f)
        logger.info(f"Loaded {len(data_items)} questions from input file")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
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
    
   
    predictions = {}
    
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("uid", f"item-{i}")
        
        logger.info(f"Processing item {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        question = item.get("question", "")
        grouth_truth = item.get("answer", "")
        question_type = item.get("question_type", "")
        
        prompt = create_prompt_from_multhiertt(item)
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
 
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
            
          
            if model_type in ["deepseek-r1", "deepseek-r1-inner"]:
              
                success, answer, thinking = api_result
            else:
              
                success, answer = api_result

            if not success:
                raise Exception(f"API call failed: {answer}")
            
          
            call_time = time.time() - call_start
            
          
            token_info = {}
            if model_type == "openai" and hasattr(answer, 'usage'):
                token_info = {
                    "completion_tokens": getattr(answer.usage, 'completion_tokens', 'N/A'),
                    "prompt_tokens": getattr(answer.usage, 'prompt_tokens', 'N/A'),
                    "total_tokens": getattr(answer.usage, 'total_tokens', 'N/A')
                }
                
                answer = answer.choices[0].message.content
            else:
                token_info = {"note": "This model type does not provide token usage statistics"}
            
          
            extracted_answer = extract_answer_from_response(answer)
            
           
            result = {
                "uid": item_id,
                "question": question,
                "prompt": prompt,
                "question_type": question_type,
                "ground_truth": grouth_truth,
                "model_answer": answer,
                "extracted_answer": extracted_answer,
                "processing_time": call_time,
                "token_usage": token_info
            }
            
           
            if thinking is not None:
                result["reasoning"] = thinking
            
            results.append(result)
            success_count += 1
            
           
            predictions[item_id] = extracted_answer
            
         
            logger.info(f"Question: {question}")
            logger.info(f"Question type: {question_type}")
            logger.info(f"Ground truth: {grouth_truth}")
            logger.info(f"Model answer: {extracted_answer}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing item {i+1}: {e}")
   
            result = {
                "uid": item_id,
                "question": question,
                "model_answer": f"Processing error: {str(e)}",
                "error": str(e)
            }
            results.append(result)
            
        
            predictions[item_id] = ""
        
  
        if (i + 1) % 5 == 0 or error_count > 0:
            try:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(data_items)})")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
    
  
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Detailed results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save detailed results file: {e}")
    

    eval_output_file = output_file.replace('.json', '_eval.json')
    try:
      
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


    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Processing failures: {error_count}/{len(data_items)}")
    logger.info("=" * 60)

def main():
    # 检测正确的根路径
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
    
    # 参数设置
    input_file = os.path.join(base_path, "data/MultiHiertt/processed_test-dev_out_3_8.json")  # MultHier-TT测试集
    output_file = os.path.join(base_path, "results/multihiertt/multihiertt_sft_results.json")
    model_name = "/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"  # 可以使用"deepseek-r1"或本地模型路径 /mnt/usercache/huggingface/Qwen2.5-7B-Instruct
    log_file = os.path.join(base_path, "results/multihiertt/logs/multihiertt_sft_processing.log")
    max_tokens = 3096  # 最大输出token数
    temperature = 0.6  
    
    # 检查点参数 - 从哪个数据项开始（0表示从头开始）
    start_from = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    process_multhiertt_data(input_file, output_file, model_name, log_file, max_tokens, temperature, start_from)

if __name__ == "__main__":
    main()