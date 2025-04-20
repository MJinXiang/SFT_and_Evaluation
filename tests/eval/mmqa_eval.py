import json
import os
import re
import argparse
import sys
import math
from typing import Dict, List, Union, Any, Tuple, Set
from collections import Counter

def extract_answer_from_response(model_answer: str) -> Any:

    try:
        # 尝试找到JSON格式的回答
        json_pattern = r'\{.*"Answers"\s*:\s*(.*?)\s*\}'
        json_match = re.search(json_pattern, model_answer, re.DOTALL | re.IGNORECASE)
        
        if json_match:
            json_content = "{\"Answers\": " + json_match.group(1) + "}"
            try:
                parsed_json = json.loads(json_content)
                return parsed_json.get("Answers")
            except:
                pass
        
        # 尝试解析整个回答为JSON
        try:
            parsed = json.loads(model_answer)
            if "Answers" in parsed:
                return parsed["Answers"]
        except:
            pass
            
        # 如果上述方法都失败，尝试正则表达式提取数值答案
        number_match = re.search(r'\b(\d+)\b', model_answer)
        if number_match:
            return int(number_match.group(1))
            
        return model_answer.strip()
    except Exception as e:
        print(f"Error extracting answer: {e}")
        return None

def normalize_value(value):
    """标准化值，处理不同类型的数据"""
    # 如果是数字类型，转换为字符串，保留两位小数
    if isinstance(value, (int, float)):
        # 处理百分比
        if value <= 1 and value > 0 and ('percent' in str(value).lower() or '%' in str(value)):
            return f"{value * 100:.2f}%"
        return f"{value:.2f}" if isinstance(value, float) else str(value)
    
    # 如果是字符串，进行标准化处理
    if isinstance(value, str):
        # 转换为小写
        value = value.lower()
        # 移除前后空格
        value = value.strip()
        # 移除美元符号、逗号等
        value = value.replace('$', '').replace(',', '')
        # 处理百分比
        if '%' in value:
            try:
                num_value = float(value.replace('%', ''))
                return f"{num_value:.2f}%"
            except:
                pass
        
        # 处理数值型字符串
        try:
            num = float(value)
            return f"{num:.2f}" if num != int(num) else str(int(num))
        except:
            pass
            
    # 其他情况返回原始字符串的小写形式
    return str(value).lower().strip()

def check_answer_correctness(model_answer: str, expected_answer: Dict[str, Any]) -> Tuple[bool, bool, Any]:
    """检查模型回答是否正确
    
    对于MMQA任务，比较预测答案与预期答案是否匹配，忽略顺序差异
    
    Returns:
        Tuple[bool, bool, Any]: (完全匹配标志, 部分匹配标志, 提取的答案)
    """
    extracted_answer = extract_answer_from_response(model_answer)
    if extracted_answer is None:
        return False, False, None
    
    # 只获取期望答案的data部分
    expected_data = expected_answer.get("data", [])
    
    # 浮点数比较函数 - 只比较小数点后两位
    def float_equal(a, b, decimal_places=2):
        try:
            # 转换为浮点数
            a_float = float(a)
            b_float = float(b)
            
            # 四舍五入到指定小数位数
            a_rounded = round(a_float, decimal_places)
            b_rounded = round(b_float, decimal_places)
            
            # 直接比较四舍五入后的值
            return a_rounded == b_rounded
        except (ValueError, TypeError):
            # 如果不能转换为浮点数，则进行字符串比较
            return normalize_value(a) == normalize_value(b)
    
    # 将数据标准化为便于比较的格式
    def normalize_data(data):
        if not isinstance(data, list):
            return set([normalize_value(data)])
        
        if not data:
            return set()
            
        # 处理嵌套列表
        if isinstance(data[0], list):
            result = set()
            for sublist in data:
                for item in sublist:
                    result.add(normalize_value(item))
            return result
        else:
            # 处理平面列表
            return set(normalize_value(item) for item in data)
    
    # 将预期答案和提取的答案转换为标准化的集合
    expected_set = normalize_data(expected_data)
    extracted_set = normalize_data(extracted_answer)
    
    # 完全匹配 - 两个集合完全相同
    exact_match = (expected_set == extracted_set)
    
    # 部分匹配 - 交集不为空且至少包含50%的预期元素
    intersection = expected_set.intersection(extracted_set)
    partial_match = False
    
    if intersection:
        # 计算交集元素占预期元素的比例
        coverage_ratio = len(intersection) / len(expected_set) if expected_set else 0
        # 如果覆盖率超过50%，则视为部分匹配
        partial_match = coverage_ratio >= 0.8
    
    return exact_match, partial_match, extracted_answer

def evaluate_mmqa_results(input_file, output_file=None):
    """评估MMQA任务的预测结果"""
    try:
        # 加载输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        print(f"加载了 {len(data)} 个预测结果进行评估")
        
        # 初始化统计数据
        total_examples = len(data)
        exact_match_count = 0
        partial_match_count = 0
        results = []
        
        error_items = []  # 用于记录处理有问题的项
        skipped_items = []  # 用于记录由于提示过长而跳过的项
        
        # 评估每个预测结果
        for i, item in enumerate(data):
            try:
                # 获取必要字段
                item_id = item.get('uid', f'unknown-{i}')
                model_answer = item.get('model_answer', '')
                extracted_answer = item.get('extracted_answer', model_answer)
                
                # 检查是否是因提示过长而跳过的样本
                if "Skipped due to excessive prompt length" in model_answer:
                    print(f"样本 {item_id} 因提示过长而被跳过，不计入评估")
                    skipped_items.append(item_id)
                    continue
                
                # 获取真实答案
                ground_truth = item.get('ground_truth')
                if not ground_truth:
                    print(f"示例 {item_id} 没有找到真实答案")
                    continue
                
                # 检查答案正确性
                exact_match, partial_match, extracted = check_answer_correctness(
                    extracted_answer, ground_truth)
                    
                # 更新统计数据
                if exact_match:
                    exact_match_count += 1
                if partial_match:
                    partial_match_count += 1
                    
                # 详细显示评估进度（每10个样本）
                if (i + 1) % 10 == 0:
                    print(f"已评估: {i+1}/{total_examples}, EM: {exact_match_count}, PM: {partial_match_count}")
                
                # 记录评估结果
                evaluation_result = {
                    'uid': item_id,
                    'question': item.get('question', ''),
                    'prediction': extracted,
                    'ground_truth': ground_truth.get('data', []),
                    'exact_match': exact_match,
                    'partial_match': partial_match,
                    'model_answer': model_answer[:200] + ('...' if len(model_answer) > 200 else '')
                }
                results.append(evaluation_result)
            
            except Exception as e:
                print(f"处理样本 {item.get('uid', i)} 时出错: {e}")
                error_items.append(item.get('uid', i))
                # 继续处理下一个样本
                continue
        
        # 计算有效样本数（总数减去跳过的样本）
        valid_examples = total_examples - len(skipped_items)
        
        # 计算总体指标 - 使用有效样本数作为分母
        exact_match_ratio = exact_match_count / valid_examples if valid_examples > 0 else 0
        partial_match_ratio = partial_match_count / valid_examples if valid_examples > 0 else 0
        
        # 打印评估结果摘要
        print("\n" + "=" * 50)
        print("MMQA 评估结果摘要")
        print("=" * 50)
        print(f"总样本数: {total_examples}")
        print(f"因提示过长而跳过的样本数: {len(skipped_items)}")
        print(f"有效评估样本数: {valid_examples}")
        print(f"成功评估样本数: {len(results)}")
        print(f"处理失败样本数: {len(error_items)}")
        if skipped_items:
            print(f"跳过样本 ID: {', '.join(str(item) for item in skipped_items[:5])}" + 
                  ("..." if len(skipped_items) > 5 else ""))
        if error_items:
            print(f"失败样本 ID: {', '.join(str(item) for item in error_items[:5])}" + 
                  ("..." if len(error_items) > 5 else ""))
        print(f"完全匹配 (EM): {exact_match_count}/{valid_examples} = {exact_match_ratio:.4f}")
        print(f"部分匹配 (PM): {partial_match_count}/{valid_examples} = {partial_match_ratio:.4f}")
        print("=" * 50)
        
        # 按问题类型分析结果
        question_type_stats = {}
        for item in results:
            q_type = item.get('question_type', 'unknown')
            if q_type not in question_type_stats:
                question_type_stats[q_type] = {
                    'total': 0, 'exact_match': 0, 'partial_match': 0
                }
            
            question_type_stats[q_type]['total'] += 1
            if item['exact_match']:
                question_type_stats[q_type]['exact_match'] += 1
            if item['partial_match']:
                question_type_stats[q_type]['partial_match'] += 1
        
        # 打印按问题类型的统计数据
        if len(question_type_stats) > 1:  # 如果有多种问题类型
            print("\n按问题类型的评估结果:")
            for q_type, stats in question_type_stats.items():
                if q_type != 'unknown' and stats['total'] > 0:
                    em_ratio = stats['exact_match'] / stats['total']
                    pm_ratio = stats['partial_match'] / stats['total']
                    print(f"问题类型 '{q_type}': EM={em_ratio:.4f}, PM={pm_ratio:.4f} (样本数={stats['total']})")
        
        # 保存详细评估结果
        if output_file:
            evaluation_results = {
                'metrics': {
                    'exact_match': exact_match_ratio,
                    'partial_match': partial_match_ratio,
                    'total_examples': total_examples,
                    'skipped_examples': len(skipped_items),
                    'valid_examples': valid_examples,
                    'processed_examples': len(results),
                    'error_examples': len(error_items)
                },
                'question_type_stats': question_type_stats,
                'results': results,
                'error_items': error_items,
                'skipped_items': skipped_items
            }
            
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # 保存评估结果
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
                
            print(f"详细评估结果已保存至 {output_file}")
        
        # 返回总体评估指标
        return exact_match_ratio, partial_match_ratio
        
    except Exception as e:
        print(f"评估过程中发生错误: {e}")
        import traceback
        print(traceback.format_exc())
        return 0, 0

# def main():
#     parser = argparse.ArgumentParser(description='评估MMQA任务的预测结果')
#     parser.add_argument('--input', type=str, default='/netcache/mengjinxiang/Project/LLaMA-Factory-main/results/mmqa/mmqa2qa_3b_results.json',
#                         help='包含预测结果和真实答案的输入文件路径')
#     parser.add_argument('--output', type=str, default='/netcache/mengjinxiang/Project/LLaMA-Factory-main/baseline/eval/eval_results/mmqa2qa_3b_results.json',
#                         help='评估结果输出文件路径')
    
#     args = parser.parse_args()
    
#     print(f"开始评估MMQA预测结果: {args.input}")
#     evaluate_mmqa_results(args.input, args.output)
#     print("评估完成！")

def main():
    parser = argparse.ArgumentParser(description='Evaluate MMQA prediction results')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to input file containing predictions and ground truth')
    parser.add_argument('--output_file', type=str, required=True,
                        help='Path to save evaluation results')
    parser.add_argument('--base_path', type=str,
                        help='Base path for the project (optional)')
    
    args = parser.parse_args()
    
    print(f"Starting evaluation of MMQA predictions: {args.results_file}")
    evaluate_mmqa_results(args.results_file, args.output_file)
    print("Evaluation completed!")

if __name__ == '__main__':
    main()