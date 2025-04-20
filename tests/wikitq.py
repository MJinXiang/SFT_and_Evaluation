
import json
import os
import logging
import time
import random
import sys
import argparse
from datetime import datetime
from openai import OpenAI
from prompt import COT_PROMPT_TEMPLATE
from llm import initialize_client, call_api_with_retry  # type: ignore

# 设置日志
def setup_logger(log_file):
    """Set up the logger"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('table_qa_processor')
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

def create_prompt_from_wikitq(item):
    table_str = ""
    if "table" in item:
        # 创建表格的字符串表示
        for row in item["table"]:
            table_str += " | ".join([str(cell) for cell in row]) + "\n"
    
    prompt = COT_PROMPT_TEMPLATE.format(table=table_str, question=item["question"])
    
    return prompt

def process_table_qa_data(input_file, output_file, model_name, log_file, max_tokens=2048, start_from=0, api_port=8000, temperature=0.0):
    logger = setup_logger(log_file)
    
    # 记录开始时间
    start_time = time.time()
    logger.info(f"Started processing table QA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Using model: {model_name}")
    logger.info(f"API port: {api_port}")
    logger.info(f"Starting from index {start_from}")
    
    # 初始化模型客户端
    try:
        client_info = initialize_client({"model_path": model_name, "api_port": api_port})
        logger.info(f"Model client initialized successfully, type: {client_info['model_type']}")
    except Exception as e:
        logger.error(f"Model client initialization failed: {e}")
        return
    
    # 读取JSONL文件
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
    
    # 检查是否存在中间结果，如果有则加载
    results = []
    if start_from > 0 and os.path.exists(f"{output_file}.temp"):
        try:
            with open(f"{output_file}.temp", 'r', encoding='utf-8') as f:
                results = json.load(f)
            logger.info(f"Loaded intermediate results, containing {len(results)} records")
        except Exception as e:
            logger.error(f"Failed to load intermediate results: {e}, starting from scratch")
            start_from = 0
    
    success_count = len(results)
    error_count = 0
    
    # 处理每个数据项
    for i, item in enumerate(data_items[start_from:], start=start_from):
        item_id = item.get("id", f"item-{i}")
        logger.info(f"Processing data item {i+1}/{len(data_items)}... [ID: {item_id}]")
        
        # 创建针对WikiTableQuestion的提示
        prompt = create_prompt_from_wikitq(item)
        
        # 提取数据项中的prompt作为用户消息
        messages = [{"role": "user", "content": prompt}]
        
        try:
            # 记录API调用开始时间
            call_start = time.time()
            
            # 调用API
            api_result = call_api_with_retry(
                client_info=client_info,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0,
                max_retries=10
            )
            
            # 初始化thinking变量，避免未定义错误
            thinking = None
            
            # 处理API结果 - 检查是否是deepseek-r1模型
            if client_info["model_type"] in ["deepseek-r1", "deepseek-r1-inner"]:
                # deepseek-r1返回结果包含三个值：成功标志、回答内容和推理内容
                success, answer, thinking = api_result
            else:
                # 其他模型返回两个值：成功标志和回答内容
                success, answer = api_result

            if not success:
                raise Exception(f"API调用失败: {answer}")
            
            # 计算API调用时间
            call_time = time.time() - call_start
            
            # 处理返回的响应 - 提取token使用情况
            token_info = {}
            if client_info["model_type"] == "openai" and hasattr(answer, 'usage'):
                token_info = {
                    "completion_tokens": getattr(answer.usage, 'completion_tokens', 'N/A'),
                    "prompt_tokens": getattr(answer.usage, 'prompt_tokens', 'N/A'),
                    "total_tokens": getattr(answer.usage, 'total_tokens', 'N/A')
                }
                # 提取回答内容
                answer = answer.choices[0].message.content
            else:
                token_info = {"note": "Token usage not available for this model type"}
            
            # 构建结果对象
            result = {
                "id": item_id,
                "source": item.get("source", {}),
                "prompt": prompt,
                "question": item["question"],
                "answer": item["answer"],
                "model_answer": answer,
                "processing_time": call_time,
                "token_usage": token_info
            }
            
            # 添加思考内容字段（针对deepseek-r1模型）
            if thinking is not None:
                result["think"] = thinking
                logger.info(f"Model thinking: {thinking}")
            
            results.append(result)
            success_count += 1
            
            # 记录详细日志
            logger.info(f"Question: {item['question']}")
            logger.info(f"Golden answer: {item['answer']}")
            logger.info(f"Model answer: {answer}")
            logger.info(f"Processing time: {call_time:.2f} seconds")
            logger.info(f"Token usage: {token_info}")
            logger.info("-" * 50)
            
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing data item {i+1}: {e}")
            # 记录错误信息
            result = {
                "id": item_id,
                "source": item.get("source", {}),
                "question": item["question"],
                "answer": item["answer"],
                "model_answer": f"Processing error: {str(e)}",
                "error": str(e)
            }
            results.append(result)
        
        # 每处理5个数据项或出错时保存中间结果
        if (i + 1) % 5 == 0 or error_count > 0:
            try:
                with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved intermediate results ({i+1}/{len(data_items)})")
            except Exception as e:
                logger.error(f"Failed to save intermediate results: {e}")
    
    # 将结果保存到JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")

    # 记录总结信息
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing complete! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}")
    logger.info(f"Failed to process: {error_count}/{len(data_items)}")
    if len(data_items) > 0:
        logger.info(f"Success rate: {success_count/len(data_items)*100:.2f}%")
    logger.info("=" * 60)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process WikiTableQuestions dataset with LLM')
    
    parser.add_argument('--api_port', type=int, default=8000, help='API port for local model server')
    parser.add_argument('--output_file', type=str, help='Path to save results')
    parser.add_argument('--model_path', type=str, help='Model path or identifier')
    parser.add_argument('--log_file', type=str, help='Path to log file')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for model generation')
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
    input_file = os.path.join(base_path, "data/wikitq/test.jsonl")
    
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
    process_table_qa_data(input_file, output_file, model_name, log_file, max_tokens, start_from, api_port, temperature)


if __name__ == "__main__":
    main()