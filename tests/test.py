#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# filename: test_api_call.py

import os
import json
import time
import logging
import argparse
from datetime import datetime
from utils.llm import call_api_with_retry, initialize_client

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"api_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger(__name__)

# 示例评估提示模板
EVAL_PROMPT_TEMPLATE = """Here is the original question, the correct answer, and the candidate answer. Please evaluate whether the correct answer and the candidate answer are consistent.

# Examples:

Question: What is the capital of France?
Candidate Answer: The capital of France is Paris
Correct Answer: Paris is the capital of France
Consistent: Yes
---------------

Question: What is the distance from Paris to London?
Candidate Answer: 5 km
Correct Answer: 10 km
Consistent: No
--------------

# YOUR TASK

Respond with only "Yes" or "No" (without quotes). Do not include a rationale.

Question: {question}
Candidate Answer: {candidate_answer}
Correct Answer: {correct_answer}
Consistent:"""


def test_api_call(model_name, concurrency=1, sample_count=5, max_retries=3):
    """
    测试API调用模型
    
    Args:
        model_name: API模型名称
        concurrency: 并发数量
        sample_count: 测试样本数量
        max_retries: 最大重试次数
    """
    logger.info(f"开始测试API调用: {model_name}, 并发数={concurrency}, 样本数={sample_count}")
    
    # 初始化客户端
    try:
        client_info = initialize_client({"model_path": model_name})
        logger.info(f"已初始化API客户端: {client_info['model_type']}")
    except Exception as e:
        logger.error(f"初始化客户端失败: {e}")
        return False
    
    # 准备测试样本
    test_samples = [
        {
            "question": "What is the capital of France?",
            "correct_answer": "Paris",
            "candidate_answer": "The capital of France is Paris"
        },
        {
            "question": "What is the population of Tokyo?",
            "correct_answer": "13.96 million",
            "candidate_answer": "About 14 million people"
        },
        {
            "question": "When was the Declaration of Independence signed?",
            "correct_answer": "July 4, 1776",
            "candidate_answer": "July 4th, 1776"
        },
        {
            "question": "What is the distance from Earth to the Sun?",
            "correct_answer": "93 million miles",
            "candidate_answer": "150 million kilometers"
        },
        {
            "question": "Who wrote Hamlet?",
            "correct_answer": "William Shakespeare",
            "candidate_answer": "Shakespeare"
        },
        {
            "question": "What is the square root of 64?",
            "correct_answer": "8",
            "candidate_answer": "8.0"
        },
        {
            "question": "What is the boiling point of water?",
            "correct_answer": "100°C",
            "candidate_answer": "212°F"
        },
        {
            "question": "Who was the first president of the United States?",
            "correct_answer": "George Washington",
            "candidate_answer": "Washington was the first US president"
        },
        {
            "question": "What is the chemical formula for water?",
            "correct_answer": "H2O",
            "candidate_answer": "Water is H2O"
        },
        {
            "question": "How many planets are in our solar system?",
            "correct_answer": "8",
            "candidate_answer": "There are 8 planets"
        }
    ]
    
    # 限制样本数量
    test_samples = test_samples[:min(sample_count, len(test_samples))]
    
    # 串行测试
    if concurrency <= 1:
        logger.info("开始串行API调用测试")
        for i, sample in enumerate(test_samples):
            prompt = EVAL_PROMPT_TEMPLATE.format(
                question=sample["question"],
                candidate_answer=sample["candidate_answer"],
                correct_answer=sample["correct_answer"]
            )
            
            start_time = time.time()
            success, response = call_api_with_retry(
                client_info,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
                max_retries=max_retries
            )
            elapsed_time = time.time() - start_time
            
            if success:
                logger.info(f"样本 {i+1}: 成功 - 耗时: {elapsed_time:.2f}秒")
                logger.info(f"  问题: {sample['question']}")
                logger.info(f"  回答: {response.strip()}")
            else:
                logger.error(f"样本 {i+1}: 失败 - 耗时: {elapsed_time:.2f}秒")
                logger.error(f"  错误: {response}")
    
    # 并行测试
    else:
        import concurrent.futures
        
        logger.info(f"开始并行API调用测试 (并发数={concurrency})")
        
        def process_sample(idx, sample):
            prompt = EVAL_PROMPT_TEMPLATE.format(
                question=sample["question"],
                candidate_answer=sample["candidate_answer"],
                correct_answer=sample["correct_answer"]
            )
            
            start_time = time.time()
            success, response = call_api_with_retry(
                client_info,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
                max_retries=max_retries
            )
            elapsed_time = time.time() - start_time
            
            return {
                "index": idx,
                "question": sample["question"],
                "success": success,
                "response": response,
                "elapsed_time": elapsed_time
            }
        
        # 使用线程池进行并行调用
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(process_sample, i, sample): i 
                for i, sample in enumerate(test_samples)
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    if result["success"]:
                        logger.info(f"样本 {result['index']+1}: 成功 - 耗时: {result['elapsed_time']:.2f}秒")
                        logger.info(f"  问题: {result['question']}")
                        logger.info(f"  回答: {result['response'].strip()}")
                    else:
                        logger.error(f"样本 {result['index']+1}: 失败 - 耗时: {result['elapsed_time']:.2f}秒")
                        logger.error(f"  错误: {result['response']}")
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"样本 {idx+1} 处理出错: {e}")
    
    logger.info("API调用测试完成")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description='测试API模型调用')
    parser.add_argument('--model', type=str, default='claude-3-7-sonnet-20250219',
                        help='API模型名称 (默认: claude-3-7-sonnet-20250219)')
    parser.add_argument('--concurrency', type=int, default=5,
                        help='并发数量 (默认: 5)')
    parser.add_argument('--samples', type=int, default=10,
                        help='测试样本数量 (默认: 10)')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='最大重试次数 (默认: 3)')
    parser.add_argument('--debug', action='store_true',
                        help='启用调试模式，输出详细日志')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
        logging.getLogger("model_caller").setLevel(logging.DEBUG)
    
    # 检查环境变量
    api_key_vars = [
        "ANTHROPIC_API_KEY",      # Claude API密钥
        "DASHSCOPE_API_KEY",      # 灵积API密钥
        "DASHSCOPE_FANGYU_API_KEY", # 灵积方舟API密钥
        "OPENAI_API_KEY",         # OpenAI API密钥
        "BAICHUAN_API_KEY",       # 百川API密钥
        "GEMINI_API_KEY"          # Gemini API密钥
    ]
    
    missing_keys = []
    for key_var in api_key_vars:
        if not os.environ.get(key_var):
            missing_keys.append(key_var)
    
    if missing_keys:
        logger.warning(f"以下API密钥环境变量未设置: {', '.join(missing_keys)}")
        logger.warning("某些API调用可能会失败，请设置相应的环境变量")
    
    # 运行测试
    success = test_api_call(
        args.model, 
        concurrency=args.concurrency,
        sample_count=args.samples,
        max_retries=args.max_retries
    )
    
    if success:
        logger.info("API测试完成")
        exit(0)
    else:
        logger.error("API测试失败")
        exit(1)

# import os
# import json
# import time
# import requests

# def inference2(prompt, max_tokens=4000, temperature=1):
#     """Make an inference request using ByteIntl API"""
#     API_KEY = "54nhP5uBXv7iWgHJ4bWMD90Nwkn09BXN"
#     API_URL = "https://gpt-i18n.byteintl.net/gpt/openapi/online/v2/crawl"
    
#     headers = {
#         "Content-Type": "application/json",
#         "X-TT-LOGID": "asdfasfadsfa"  # 使用您示例中的logid，实际使用中可能需要动态生成
#     }
    
#     # 构建请求数据
#     request_data = {
#         "model": "gcp-claude37-sonnet",
#         "max_tokens": max_tokens,
#         "messages": [
#             {
#                 "content": prompt,
#                 "role": "user"
#             }
#         ],
#         "thinking": {
#             "type": "enabled",
#             "budget_tokens": 2000
#         },
#         "stream": False,
#         "temperature": temperature
#     }
    
#     # URL参数中添加ak
#     url_with_params = f"{API_URL}?ak={API_KEY}"
    
#     # 发送请求，加入重试逻辑
#     for retry in range(3):
#         try:
#             response = requests.post(url_with_params, headers=headers, data=json.dumps(request_data))
#             if response.status_code == 200:
#                 response_json = response.json()
#                 if "choices" in response_json and len(response_json["choices"]) > 0:
#                     model_response = response_json["choices"][0]["message"]["content"]
#                     return model_response
#                 else:
#                     print(f"Unexpected response format: {response_json}")
            
#             error_msg = f"API request error: {response.status_code}, {response.text}"
#             print(f"{error_msg} - Retrying ({retry+1}/3)")
#             if  response.json()["error"]["code"]=='-4003':
#                 return None
#             time.sleep(10.0)
        
#         except Exception as e:
#             print(f"API request exception: {str(e)} - Retrying ({retry+1}/3)")
#             time.sleep(30)
    
#     return None


# prompt = "What is the capital of France?"
# response = inference2(prompt)
# print(f"Response: {response}")