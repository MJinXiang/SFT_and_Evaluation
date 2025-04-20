#### python
# filepath: /netcache/mengjinxiang/Project/LLaMA-Factory-main/baseline/eval/hybridqa_eval.py

import json
import re
import collections
import string
import sys
import argparse
import os


def clean_answer_tags(answer_text):
 
    if not answer_text:
        return ""
    
    # 移除<answer>和</answer>标签
    cleaned_text = re.sub(r'</?answer>', '', answer_text)
    
    # 检查是否包含思考块
    think_pattern = r'<think>(.*?)</think>\s*'
    think_match = re.search(think_pattern, cleaned_text, re.DOTALL)
    
    # 如果找到思考块
    if think_match:
        # 获取思考块内部文本
        think_content = think_match.group(1)
        # 提取思考块后面的内容
        after_think_content = cleaned_text[think_match.end():].strip()
        
        # 如果思考块后面有内容，使用后面的内容
        if after_think_content:
            cleaned_text = after_think_content
        else:
            # 如果思考块后面没有内容，从思考块内部寻找答案
            answer_pattern = r'(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)'
            answer_match = re.search(answer_pattern, think_content, re.IGNORECASE)
            
            if answer_match:
                # 从思考块内部提取答案
                extracted_answer = answer_match.group(1).strip()
                # 处理可能的引号和句号
                extracted_answer = re.sub(r'^["\'](.*?)["\']\.?$', r'\1', extracted_answer)
                return extracted_answer
            
            # 如果思考块内没有明确的"the answer is"，尝试匹配"Therefore, ..."或"Thus, ..."
            conclusion_pattern = r'(?:therefore|thus|so|hence|consequently|in conclusion)[,:]?\s+(.*?)(?:$|\.|\n)'
            conclusion_match = re.search(conclusion_pattern, think_content, re.IGNORECASE)
            
            if conclusion_match:
                return conclusion_match.group(1).strip()
    
    # 在完整文本中查找"the answer is"模式
    answer_pattern = r'(?:the\s+answer\s+is\s*:?\s*)(.*?)(?:$|\.|\n)'
    answer_match = re.search(answer_pattern, cleaned_text, re.IGNORECASE)
    
    if answer_match:
        # 提取匹配的内容
        extracted_answer = answer_match.group(1).strip()
        # 处理可能的引号
        extracted_answer = re.sub(r'^["\'](.*)["\']$', r'\1', extracted_answer)
        return extracted_answer
    
    # 如果没有找到特定模式，返回清理后的文本
    return cleaned_text.strip()

def normalize_answer(s):
    """规范化答案文本：转为小写、去除标点符号、冠词和多余空格"""
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    """获取文本的标准化token"""
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    """计算精确匹配分数"""
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    """计算F1分数"""
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # 如果答案为空，F1为1（如果两者都为空），否则为0
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def prepare_model_results(model_results_file):
    """准备模型结果数据，提取提取的答案"""
    with open(model_results_file, 'r', encoding='utf-8') as f:
        model_results = json.load(f)
    
    # 构建问题ID到预测答案的映射
    results = []
    for item in model_results:
        extracted_answer = item.get("extracted_answer", "")
        cleaned_answer = clean_answer_tags(extracted_answer)
        
        result = {
            "question_id": item["question_id"],
            "question": item.get("question", ""),
            "pred": cleaned_answer  # 使用提取的答案作为预测
        }
        results.append(result)
    
    return results

def prepare_test_answers(test_answers_file):
    """准备测试答案数据"""
    with open(test_answers_file, 'r', encoding='utf-8') as f:
        test_answers = json.load(f)
    
    # 构建问题ID到黄金答案的映射
    reference = {"reference": {}, "table": [], "passage": []}
    
    for item in test_answers:
        question_id = item["question_id"]
        reference["reference"][question_id] = item["pred"]  # 使用正确答案
        
        # 根据目标类型判断是表格还是段落
        if "target" in item and item["target"] and len(item["target"]) > 2:
            if item["target"][2] is not None:  # 如果有链接，则是passage
                reference["passage"].append(question_id)
            else:
                reference["table"].append(question_id)
    
    # 确保每个问题都被分配到表格或段落
    for qid in reference["reference"].keys():
        if qid not in reference["table"] and qid not in reference["passage"]:
            reference["table"].append(qid)  # 默认分配到表格类型
    
    return reference


def get_raw_scores(model_results, reference):
    """计算精确匹配和F1分数，只计算在模型结果和测试答案中都存在的问题ID"""
    exact_scores = {}
    f1_scores = {}
    
    # 找出在模型结果中实际存在的问题ID
    evaluated_qids = set(example['question_id'] for example in model_results)
    
    # 按照实际评估的问题ID过滤参考答案
    filtered_reference = {
        "reference": {k: v for k, v in reference['reference'].items() if k in evaluated_qids},
        "table": [k for k in reference['table'] if k in evaluated_qids],
        "passage": [k for k in reference['passage'] if k in evaluated_qids]
    }
    
    # 计算分数
    for example in model_results:
        qas_id = example['question_id']
        if qas_id in filtered_reference['reference']:
            gold_answer = filtered_reference['reference'][qas_id]
            prediction = example['pred']
            
            exact_scores[qas_id] = compute_exact(gold_answer, prediction)
            f1_scores[qas_id] = compute_f1(gold_answer, prediction)

    # 使用过滤后的问题ID列表
    qid_list = list(filtered_reference['reference'].keys())
    total = len(qid_list)
    
    table_list = filtered_reference['table']
    passage_list = filtered_reference['passage']
    
    # 防止除零错误
    table_exact = 0
    table_f1 = 0
    if table_list:
        table_exact = 100.0 * sum(exact_scores.get(k, 0) for k in table_list) / len(table_list)
        table_f1 = 100.0 * sum(f1_scores.get(k, 0) for k in table_list) / len(table_list)
    
    passage_exact = 0
    passage_f1 = 0
    if passage_list:
        passage_exact = 100.0 * sum(exact_scores.get(k, 0) for k in passage_list) / len(passage_list)
        passage_f1 = 100.0 * sum(f1_scores.get(k, 0) for k in passage_list) / len(passage_list)

    total_exact = 100.0 * sum(exact_scores.get(k, 0) for k in qid_list) / total if total > 0 else 0
    total_f1 = 100.0 * sum(f1_scores.get(k, 0) for k in qid_list) / total if total > 0 else 0

    # 添加额外信息
    return collections.OrderedDict(
        [
            ("table exact", table_exact),
            ("table f1", table_f1),
            ("passage exact", passage_exact),
            ("passage f1", passage_f1),
            ("total exact", total_exact),
            ("total f1", total_f1),
            ("total", total),                       # 实际评估的问题总数
            ("table_count", len(table_list)),       # 表格问题数量
            ("passage_count", len(passage_list)),   # 段落问题数量
            ("total_reference", len(reference['reference'])),  # 参考答案中的总问题数
        ]
    )

def create_eval_format(model_results_file, output_file):
    """创建符合评估格式的输出文件"""
    with open(model_results_file, 'r', encoding='utf-8') as f:
        model_results = json.load(f)
    
    eval_format = []
    for item in model_results:
        eval_item = {
            "question_id": item["question_id"],
            "pred": item.get("extracted_answer", "")
        }
        eval_format.append(eval_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(eval_format, f, ensure_ascii=False, indent=2)
    
    print(f"评估格式文件已保存到: {output_file}")

#### python
# filepath: /netcache/mengjinxiang/Project/LLaMA-Factory-main/baseline/eval/hybridqa_eval.py

def evaluate_model(model_results_file, test_answers_file, evaluate_save_dir, create_eval_format_flag=True):
    """评估模型结果"""
    
    # 如果需要创建评估格式文件
    if create_eval_format_flag:
        output_file = os.path.splitext(model_results_file)[0] + "_eval_format.json"
        create_eval_format(model_results_file, output_file)
    
    # 准备数据
    model_results = prepare_model_results(model_results_file)
    reference = prepare_test_answers(test_answers_file)
    
    # 计算得分
    scores = get_raw_scores(model_results, reference)
    


    # 输出结果
    print("\n========== HybridQA评估结果 ==========")
    print(f"模型结果文件: {model_results_file}")
    print(f"测试答案文件: {test_answers_file}")
    print(f"评估保存目录: {evaluate_save_dir}")
    print(f"实际评估问题数: {scores['total']} / {scores['total_reference']}") # 显示实际评估/总问题数
    print("-" * 40)
    print(f"表格问题数量: {scores['table_count']}")
    print(f"表格问题精确匹配: {scores['table exact']:.2f}%")
    print(f"表格问题F1分数: {scores['table f1']:.2f}%")
    print(f"段落问题数量: {scores['passage_count']}")
    print(f"段落问题精确匹配: {scores['passage exact']:.2f}%")
    print(f"段落问题F1分数: {scores['passage f1']:.2f}%")
    print(f"总体精确匹配: {scores['total exact']:.2f}%")
    print(f"总体F1分数: {scores['total f1']:.2f}%")
    print("======================================\n")
    
    # 创建详细的评估结果，包括真实答案和模型回答
    detailed_results = []
    
    # 加载完整的模型结果以获取模型的详细回答
    with open(model_results_file, 'r', encoding='utf-8') as f:
        full_model_results = json.load(f)
    
    # 为了快速查找，创建模型回答的字典
    model_results_dict = {item["question_id"]: item for item in full_model_results}
    
    # 加载完整的测试答案以获取问题和真实答案
    with open(test_answers_file, 'r', encoding='utf-8') as f:
        full_test_answers = json.load(f)
    
    # 为了快速查找，创建测试答案的字典
    test_answers_dict = {item["question_id"]: item for item in full_test_answers}
    
    # 合并数据
    for qid in reference["reference"].keys():
        if qid in model_results_dict and qid in test_answers_dict:
            model_item = model_results_dict[qid]
            test_item = test_answers_dict[qid]
            
            # 判断答案是否正确
            gold_answer = reference["reference"][qid]
            model_answer = model_item.get("extracted_answer", "")
            cleaned_answer = clean_answer_tags(model_answer)
        
            is_correct = compute_exact(gold_answer, cleaned_answer)
            
            # 问题类型
            question_type = "passage" if qid in reference["passage"] else "table"
            
            # 创建详细结果项
            detail_item = {
                "question_id": qid,
                "question": test_item.get("question", ""),
                "question_type": question_type,
                "gold_answer": gold_answer,
                "model_answer": cleaned_answer,
                "model_full_response": model_item.get("model_answer", ""),
                "is_correct": is_correct,
                "exact_match": is_correct,
                "f1_score": compute_f1(gold_answer, cleaned_answer)
            }
            
            # 添加目标信息（如果有）
            if "target" in test_item:
                detail_item["target"] = test_item["target"]
            
            detailed_results.append(detail_item)
    
    # 创建最终的评估结果，包括摘要和详细结果
    final_result = {
        "summary": scores,
        "detailed_results": detailed_results
    }
    
    # 确保评估保存目录的父目录存在
    os.makedirs(os.path.dirname(evaluate_save_dir), exist_ok=True)
    
    # 保存结果到文件
    with open(evaluate_save_dir, 'w', encoding='utf-8') as f:
        json.dump(final_result, f, ensure_ascii=False, indent=2)
    
    print(f"详细评估结果已保存到: {evaluate_save_dir}")
    
    return scores

# def main():
#     # 使用预设的文件路径，不需要命令行参数
#     base_dir = "/netcache/mengjinxiang/Project/LLaMA-Factory-main"
    
#     # 模型结果文件
#     model_results_file = os.path.join(base_dir, "results/hybridqa/hybridqa_sft_ppo_results.json") #不同类型需要改动
    
#     # 测试答案文件
#     test_answers_file = os.path.join(base_dir, "data/hybridqa/test_answers.json")  #不用改动

#     evaluate_save_dir = os.path.join(base_dir, "baseline/eval/eval_results/hybridqa_sft_ppo_evaluation.json")  #不同类型需要改动
    
#     if not os.path.exists(model_results_file):
#         print(f"错误: 模型结果文件不存在: {model_results_file}")
#         sys.exit(1)
    
#     if not os.path.exists(test_answers_file):
#         print(f"错误: 测试答案文件不存在: {test_answers_file}")
#         sys.exit(1)

    
#     # 评估模型结果
#     evaluate_model(model_results_file, test_answers_file, evaluate_save_dir)

def main():
    parser = argparse.ArgumentParser(description='Evaluate HybridQA predictions')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to model prediction results file')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to save evaluation results')
    parser.add_argument('--base_path', type=str,
                       help='Base path for the project (default: /netcache/mengjinxiang/Project/LLaMA-Factory-main)')
    parser.add_argument('--test_data', type=str, 
                       default='data/hybridqa/test_answers.json',
                       help='Path to test answers file (default maintained for reference)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Model results file does not exist: {args.results_file}")
        sys.exit(1)
    
    if not os.path.exists(args.test_data):
        print(f"Error: Test answers file does not exist: {args.test_data}")
        sys.exit(1)

    test_path = os.path.join(args.base_path, args.test_data) 
    
    # Evaluate model results
    evaluate_model(args.results_file, test_path, args.output_file)

if __name__ == "__main__":
    main()