import json
import re
import os
import argparse
from datetime import datetime


def extract_predicted_answer(model_answer):

    if not model_answer:
        return None
    
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    answer_tag_match = re.search(answer_tag_pattern, model_answer, re.DOTALL)
    if answer_tag_match:
        model_answer = answer_tag_match.group(1).strip()
       

    match = re.search(r'Answer:\s*(.+?)(?:\n|$|\.|")', model_answer)
    if match:
        return match.group(1).strip()
    
    return None

def normalize_simple(answer):

    if not answer:
        return ""
    
 
    answer = str(answer).strip().lower()
 
    try:

        num = float(answer)

        if num.is_integer():
            return str(int(num))
        return str(num)
    except:
 
        return answer

def exact_match_simple(prediction, reference):
 
    if prediction is None or reference is None:
        return 0
    
    pred_norm = normalize_simple(prediction)
    ref_norm = normalize_simple(reference)
    
    return 1 if pred_norm == ref_norm else 0

def evaluate_answers(input_file, output_file=None, verbose=True):

    if output_file is None:
      
        dirname = os.path.dirname(input_file)
        basename = os.path.basename(input_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(dirname, f"eval_simple_{basename.split('.')[0]}_{timestamp}.json")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误: 无法解析输入文件 {input_file}，请确保它是有效的JSON。")
        return None
    
  
    if not isinstance(data, list):
        if isinstance(data, dict):
            data = [data]
        else:
            print(f"错误: 输入数据格式不正确，应为JSON对象列表或单个JSON对象。")
            return None
    
    total_samples = len(data)
    if verbose:
        print(f"开始评估 {total_samples} 个样本...")
    

    stats = {
        "total_samples": total_samples,
        "answered_samples": 0,
        "exact_matches": 0,
        "exact_match_rate": 0.0,
        "no_answer_extracted": 0,
        "samples_with_missing_truth": 0
    }
    

    results_list = []
    for i, item in enumerate(data):
 
        expected_answer = None
        if "answer" in item:
            expected_answer = item["answer"]
        elif "truth_answer" in item:
            expected_answer = item["truth_answer"] 
        elif "true_answer" in item:
            expected_answer = item["true_answer"]
        
   
        model_answer = item.get("model_answer", "")
        predicted_answer = extract_predicted_answer(model_answer)
        
 
        result_item = {
            "id": item.get("id", f"item-{i}"),
            "question": item.get("question", ""),
            "expected_answer": expected_answer,
            "predicted_answer": predicted_answer,
        }
        
      
        is_match = False
        if expected_answer is not None and predicted_answer is not None:
            is_match = exact_match_simple(predicted_answer, expected_answer)
            
            stats["answered_samples"] += 1
            stats["exact_matches"] += is_match
            result_item["is_exact_match"] = bool(is_match)
        elif expected_answer is None:
            stats["samples_with_missing_truth"] += 1
            result_item["is_exact_match"] = None
        elif predicted_answer is None:
            stats["no_answer_extracted"] += 1
            result_item["is_exact_match"] = False
        
        results_list.append(result_item)
    

    answered_samples = stats["answered_samples"]
    if answered_samples > 0:
        stats["exact_match_rate"] = stats["exact_matches"] / answered_samples
    

    evaluation_result = {
        "summary": stats,
        "results": results_list
    }
    

    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_result, f, indent=4, ensure_ascii=False)
        if verbose:
            print(f"评估结果已保存至 {output_file}")
    except Exception as e:
        if verbose:
            print(f"保存评估结果时出错: {e}")
    
    # 打印主要指标
    if verbose:
        print("\n===== 评估结果摘要 =====")
        print(f"总样本数: {stats['total_samples']}")
        print(f"有答案的样本数: {stats['answered_samples']}")
        print(f"未提取到答案的样本数: {stats['no_answer_extracted']}")
        print(f"缺少真实答案的样本数: {stats['samples_with_missing_truth']}")
        print(f"精确匹配数: {stats['exact_matches']}")
        print(f"精确匹配率(有答案的): {stats['exact_match_rate']:.4f}")
        print(f"精确匹配率(所有): {stats['exact_matches'] / stats['total_samples']:.4f}")
    
    return evaluation_result

def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description="简化版WikiTQ预测结果评估")
    parser.add_argument("--input", "-i", default='/netcache/mengjinxiang/Project/LLaMA-Factory-main/results/wikitq/wikitq_sft_ppo_results.json', help="输入文件路径（包含预测结果的JSON文件）")
    parser.add_argument("--output", "-o", default='eval_results/wikitq_sft_ppo_results.json', help="输出文件路径（评估结果将保存到此文件）")
    parser.add_argument("--quiet", "-q", action="store_true", help="静默模式，不打印输出")
    
    args = parser.parse_args()
    
    # 运行评估
    result = evaluate_answers(
        input_file=args.input, 
        output_file=args.output, 
        verbose=not args.quiet
    )
    
    if result is None:
        exit(1)

if __name__ == "__main__":
    main()