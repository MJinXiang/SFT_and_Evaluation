import json
import os
import re
import argparse
import sys
import math
from typing import Dict, List, Union, Any
from collections import Counter

def str_to_num(text):
    """将文本转换为数字，处理各种格式和清理字符"""
    if text is None:
        return "n/a"
    
    if isinstance(text, (int, float)):
        return float(text)
    
    text = str(text)
    text = text.replace("$", "").replace(",", "").replace("_", "")
    
    # 处理百分比
    if "%" in text:
        text = text.replace("%", "")
        try:
            return float(text) / 100
        except ValueError:
            return "n/a"
    
    # 处理普通数字
    try:
        return float(text)
    except ValueError:
        return "n/a"

def normalize_answer(s):
    """标准化答案文本，移除标点符号和多余空格"""
    if s is None:
        return ""
    
    # 如果是数字，返回原始格式
    if isinstance(s, (int, float)):
        return str(s)
    
    s = str(s).lower()
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'[^\w\s]', '', s)
    return s

def exact_match_score(prediction, ground_truth):
    """计算精确匹配分数"""
    if prediction is None or ground_truth is None:
        return 0
    
    # 处理数值类型的精确匹配
    pred_num = str_to_num(prediction)
    truth_num = str_to_num(ground_truth)
    
    if pred_num != "n/a" and truth_num != "n/a":
        # 使用相对容差进行比较
        if isinstance(pred_num, (int, float)) and isinstance(truth_num, (int, float)):
            # 处理0值的特殊情况
            if truth_num == 0:
                return 1.0 if abs(pred_num) < 1e-5 else 0.0
            
            # 使用相对误差
            relative_diff = abs(pred_num - truth_num) / max(abs(truth_num), 1e-10)
            return 1.0 if relative_diff < 1e-5 else 0.0
    
    # 处理文本类型的精确匹配
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0

def arithmetic_exact_match(prediction, ground_truth, tolerance=1e-5):
    """
    用于算术问题的精确匹配，允许一定的误差
    """
    pred_num = str_to_num(prediction)
    truth_num = str_to_num(ground_truth)
    
    if pred_num == "n/a" or truth_num == "n/a":
        return 0.0
    
    # 处理0值的特殊情况
    if abs(truth_num) < 1e-10:
        return 1.0 if abs(pred_num) < tolerance else 0.0
        
    # 使用相对误差或绝对误差，取较小值
    rel_tol = min(abs(truth_num) * 0.01, 0.1)  # 允许1%的相对误差，但不超过0.1
    abs_tol = min(abs(truth_num) / 1000, 0.1)  # 允许千分之一的绝对误差，但不超过0.1
    
    return 1.0 if abs(pred_num - truth_num) <= max(rel_tol, abs_tol) else 0.0

def get_tokens(s):
    """将文本分割为词元（tokens）"""
    if not s:
        return []
    s = normalize_answer(s)
    return s.split()

def compute_f1(prediction, ground_truth):
    """计算F1分数"""
    if prediction is None or ground_truth is None:
        return 0.0
    
    # 首先尝试数值比较
    pred_num = str_to_num(prediction)
    truth_num = str_to_num(ground_truth)
    
    if pred_num != "n/a" and truth_num != "n/a":
        # 如果都是数值，则使用精确匹配作为F1
        if abs(truth_num) < 1e-10:
            return 1.0 if abs(pred_num) < 1e-5 else 0.0
        
        rel_tol = min(abs(truth_num) * 0.01, 0.1)
        abs_tol = min(abs(truth_num) / 1000, 0.1)
        
        return 1.0 if abs(pred_num - truth_num) <= max(rel_tol, abs_tol) else 0.0
    
    # 文本F1计算
    prediction_tokens = get_tokens(prediction)
    ground_truth_tokens = get_tokens(ground_truth)
    
    # 如果两者都为空，返回F1=1.0
    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    
    # 计算共同词元
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    
    # 如果没有共同词元，返回F1=0
    if num_same == 0:
        return 0.0
    
    # 计算精确率和召回率
    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    
    # 计算F1
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def evaluate_predictions(input_file, output_file=None):
    """评估预测结果，从单一文件中提取预测和真实答案"""
    # 加载输入文件，该文件同时包含预测和真实答案
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 计算各项指标
    total_examples = len(data)
    exact_match_count = 0
    arithmetic_match_count = 0
    f1_score_sum = 0
    results = []
    
    for item in data:
        # 确保数据项包含必要字段
        if 'ground_truth' not in item and 'answer' not in item:
            print(f"Warning: No ground truth found for example {item.get('uid', 'unknown')}")
            continue

        # 获取真实答案
        ground_truth = item.get('ground_truth') if 'ground_truth' in item else item.get('answer')
        
        # 获取预测答案，优先使用extracted_answer
        prediction = item.get('extracted_answer')
        if not prediction:
            prediction = item.get('model_answer', '')
        
        # 计算精确匹配分数
        em_score = exact_match_score(prediction, ground_truth)
        exact_match_count += em_score
        
        # 尝试计算算术精确匹配分数
        arith_score = arithmetic_exact_match(prediction, ground_truth)
        arithmetic_match_count += arith_score
        
        # 计算F1分数
        f1 = compute_f1(prediction, ground_truth)
        f1_score_sum += f1

        prompt = item.get('prompt', '')
        question_type = item.get('question_type', '')
        
        # 保存评估结果
        result = {
            'uid': item.get('uid', 'unknown'),
            'prompt': prompt,
            'question_type': question_type,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'exact_match': em_score == 1.0,
            'arithmetic_match': arith_score == 1.0,
            'f1_score': f1
        }
        results.append(result)
    
    # 计算总体指标
    exact_match = exact_match_count / total_examples if total_examples > 0 else 0
    arithmetic_accuracy = arithmetic_match_count / total_examples if total_examples > 0 else 0
    avg_f1 = f1_score_sum / total_examples if total_examples > 0 else 0
    
    # 输出评估结果
    print(f"Total examples: {total_examples}")
    print(f"Exact Match: {exact_match:.4f}")
    print(f"Arithmetic Accuracy: {arithmetic_accuracy:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    
    # 保存详细评估结果
    if output_file:
        evaluation_results = {
            'metrics': {
                'exact_match': exact_match,
                'arithmetic_accuracy': arithmetic_accuracy,
                'f1_score': avg_f1,
                'total_examples': total_examples
            },
            'results': results
        }
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        print(f"Detailed evaluation results saved to {output_file}")
    
    return exact_match, arithmetic_accuracy, avg_f1

def main():
    parser = argparse.ArgumentParser(description='Evaluate MultHier-TT predictions')
    parser.add_argument('--input', type=str, default='/netcache/mengjinxiang/Project/LLaMA-Factory-main/results/multihiertt/multihiertt_sft_results.json', 
                        help='Path to input file containing both predictions and ground truth answers')
    parser.add_argument('--output', type=str, default='/netcache/mengjinxiang/Project/LLaMA-Factory-main/baseline/eval/eval_results/multihiertt_sft_eval_results.json', 
                        help='Path to output evaluation results')
    
    args = parser.parse_args()
    
    evaluate_predictions(args.input, args.output)

if __name__ == '__main__':
    main()