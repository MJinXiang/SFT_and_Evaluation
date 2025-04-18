import json
import re
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO
from rouge_score import rouge_scorer
from collections import defaultdict
from wrapt_timeout_decorator import timeout

# 配置常量
TIMEOUT_SECONDS = 15

@timeout(TIMEOUT_SECONDS)
def execute_with_timeout(code):
    """安全执行Python代码，带超时机制"""
    exec(code)

def extract_python_code(prediction):
    """从模型的预测中提取Python代码块"""
    pattern = r"```python\n(.*?)```"
    try:
        matches = re.findall(pattern, prediction, re.DOTALL)
        if matches:
            return matches[0]
        else:
            return ''
    except Exception as e:
        print(f"提取代码错误: {str(e)}")
        return ''

def extract_table_data(table_str):
    """从表格字符串中提取表格数据"""
    try:
        table_json = json.loads(table_str)
        table_data = []
        for item in table_json.get('data', []):
            row_data = {}
            for i in range(len(table_json.get('columns', []))):
                if i < len(item):
                    row_data[table_json['columns'][i]] = item[i]
            table_data.append(row_data)
        df = pd.DataFrame(table_data)
        return df
    except Exception as e:
        print(f"表格数据提取错误: {str(e)}")
        return pd.DataFrame()

def extract_table_from_instruction(instruction):
    """从instruction中提取表格数据"""
    pattern = r"\[TABLE\]\s*(\{.*?\})"
    match = re.search(pattern, instruction, re.DOTALL)
    if match:
        try:
            table_str = match.group(1)
            return table_str
        except:
            return None
    return None

def prepare_table_data_from_string(table_str):
    """将表格字符串保存为CSV文件以便图表代码使用"""
    try:
        df = extract_table_data(table_str)
        if not df.empty:
            df.to_csv('table.csv', index=False)
            return True
        return False
    except Exception as e:
        print(f"表格数据准备错误: {str(e)}")
        return False

# 图表数据提取函数
def get_line_y_predictions(plt_obj):
    """从线图中提取Y值数据"""
    line_y_predctions = []
    lines = plt_obj.gca().get_lines()
    line_y_predctions = [list(line.get_ydata()) for line in lines]
    return line_y_predctions

def get_bar_y_predictions(plt_obj):
    """从条形图中提取Y值数据"""
    bar_y_predctions = []
    patches = plt_obj.gca().patches
    bar_y_predctions = [patch.get_height() for patch in patches]
    return bar_y_predctions

def get_hbar_y_predictions(plt_obj):
    """从水平条形图中提取宽度值数据"""
    hbar_y_predctions = []
    patches = plt_obj.gca().patches
    hbar_y_predctions = [patch.get_width() for patch in patches]
    return hbar_y_predctions

def get_pie_y_predictions(plt_obj):
    """从饼图中提取比例数据"""
    pie_y_predctions = []
    patches = plt_obj.gca().patches
    for patch in patches:
        theta1, theta2 = patch.theta1, patch.theta2
        value = round((theta2 - theta1) / 360.0, 2)
        pie_y_predctions.append(value)
    return pie_y_predctions

def get_area_y_predictions(plt_obj):
    """从面积图中提取Y值数据"""
    area_y_predctions = []
    area_collections = plt_obj.gca().collections
    for area_collection in area_collections:
        if hasattr(area_collection, 'get_paths') and area_collection.get_paths():
            area_items = []
            for item in area_collection.get_paths()[0].vertices[:, 1]:
                if item != 0:
                    area_items.append(item)
            area_y_predctions.append(area_items)
    return list(area_y_predctions)

def get_radar_y_predictions(plt_obj):
    """从雷达图中提取数据"""
    radar_y_predctions = []
    radar_lines = plt_obj.gca().get_lines()
    radar_y_predctions = [list(line.get_ydata()) for line in radar_lines]
    for i in range(len(radar_y_predctions)):
        radar_y_predctions[i] = radar_y_predctions[i][:-1]
    return radar_y_predctions

def get_scatter_y_predictions(plt_obj):
    """从散点图中提取Y值数据"""
    scatter_y_predctions = []
    scatter_collections = plt_obj.gca().collections
    for scatter_collection in scatter_collections:
        scatter_items = []
        for item in scatter_collection.get_offsets():
            scatter_items.append(item[1])
        scatter_y_predctions.append(scatter_items)
    return scatter_y_predctions

def get_waterfall_y_predictions(plt_obj):
    """从瀑布图中提取高度数据"""
    waterfall_y_predctions = []
    patches = plt_obj.gca().patches
    waterfall_y_predctions = [patch.get_height() for patch in patches]
    return waterfall_y_predctions

# 严格比较函数
def std_digit(list_nums):
    """标准化数字列表，保留两位小数"""
    new_list = []
    for i in range(len(list_nums)):
        try:
            new_list.append(round(float(list_nums[i]), 2))
        except (ValueError, TypeError):
            new_list.append(0.0)  # 处理无法转换的值
    return new_list

def compare(list1, list2):
    """排序后严格比较两个列表"""
    list1.sort()
    list2.sort()
    if len(list1) != len(list2):
        return False
    for i in range(len(list1)):
        if np.isnan(list1[i]):
            if not np.isnan(list2[i]):
                return False
        elif list1[i] != list2[i]:
            return False
    return True

def compute_general_chart_metric(references, predictions):
    """计算一般图表的严格匹配度"""
    processed_references = []
    processed_predictions = []
    
    # 扁平化参考数据
    for reference in references:
        if isinstance(reference, list) or isinstance(reference, np.ndarray):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)
    
    # 扁平化预测数据
    for prediction in predictions:
        if isinstance(prediction, list) or isinstance(prediction, np.ndarray):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)
    
    # 标准化数据
    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)
    
    # 排序比较
    return compare(processed_references, processed_predictions)

def compute_pie_chart_metric(references, predictions):
    """计算饼图的严格匹配度"""
    processed_references = []
    processed_predictions = []
    
    # 扁平化参考数据
    for reference in references:
        if isinstance(reference, list) or isinstance(reference, np.ndarray):
            processed_references.extend(reference)
        else:
            processed_references.append(reference)
    
    references = processed_references
    processed_references = []
    
    # 转换为比例
    total = sum(references)
    if total > 0:
        for reference in references:
            processed_references.append(round(reference / total, 2))
    
    # 扁平化预测数据
    for prediction in predictions:
        if isinstance(prediction, list) or isinstance(prediction, np.ndarray):
            processed_predictions.extend(prediction)
        else:
            processed_predictions.append(prediction)
    
    # 标准化数据
    processed_references = std_digit(processed_references)
    processed_predictions = std_digit(processed_predictions)
    
    # 排序比较
    return compare(processed_references, processed_predictions)

# 相似度评估函数
def sequence_similarity(ref_seq, pred_seq):
    """计算两个数值序列的相似度"""
    if not ref_seq or not pred_seq:
        return 0.0
    
    # 转换为numpy数组
    try:
        ref_array = np.array(ref_seq, dtype=float)
        pred_array = np.array(pred_seq, dtype=float)
    except (ValueError, TypeError):
        # 处理无法转换的情况
        return 0.0
    
    # 长度惩罚
    length_ratio = min(len(ref_array), len(pred_array)) / max(len(ref_array), len(pred_array))
    
    # 处理两个序列长度不同的情况
    if len(ref_array) > len(pred_array):
        extended_pred = np.zeros_like(ref_array)
        extended_pred[:len(pred_array)] = pred_array
        pred_array = extended_pred
    elif len(pred_array) > len(ref_array):
        pred_array = pred_array[:len(ref_array)]
    
    # 计算值的相似度
    if np.max(ref_array) == np.min(ref_array):
        # 参考数据所有值相同时
        if np.max(pred_array) == np.min(pred_array):
            value_ratio = 1.0 if np.isclose(np.mean(ref_array), np.mean(pred_array), rtol=0.1) else 0.0
        else:
            value_ratio = 0.0
    else:
        # 归一化两个序列
        ref_min, ref_max = np.min(ref_array), np.max(ref_array)
        if ref_max > ref_min:
            ref_norm = (ref_array - ref_min) / (ref_max - ref_min)
            
            pred_min, pred_max = np.min(pred_array), np.max(pred_array)
            if pred_max > pred_min:
                pred_norm = (pred_array - pred_min) / (pred_max - pred_min)
                mse = np.mean((ref_norm - pred_norm) ** 2)
                value_ratio = 1.0 - min(mse, 1.0)
            else:
                value_ratio = 0.0
        else:
            value_ratio = 0.0
    
    return 0.4 * length_ratio + 0.6 * value_ratio

def compute_chart_similarity(reference, prediction):
    """计算图表数据的相似度"""
    if not prediction or not reference:
        return 0.0
    
    try:
        # 将参考和预测数据标准化为二维数组
        ref_data = reference if isinstance(reference[0], (list, np.ndarray)) else [reference]
        pred_data = prediction if isinstance(prediction[0], (list, np.ndarray)) else [prediction]
        
        # 确保两者具有相同的维度数量
        if len(ref_data) != len(pred_data):
            # 尝试找到最佳匹配
            total_score = 0
            for ref_series in ref_data:
                best_score = 0
                for pred_series in pred_data:
                    score = sequence_similarity(ref_series, pred_series)
                    best_score = max(best_score, score)
                total_score += best_score
            return total_score / len(ref_data)
        else:
            # 计算每个维度的相似度并取平均值
            scores = []
            for i in range(len(ref_data)):
                scores.append(sequence_similarity(ref_data[i], pred_data[i]))
            return sum(scores) / len(scores)
    except Exception as e:
        print(f"图表相似度计算错误: {str(e)}")
        return 0.0

def evaluate_chart_code(prediction, table_str, reference_data, chart_type):
    """评估生成的图表代码"""
    python_code = extract_python_code(prediction)
    if not python_code:
        return {
            'parsed_prediction': 'False',
            'Parse@1': False,
            'ecr_1': False,
            'similarity_score': 0.0,
            'exact_match': False
        }
    
    # 创建CSV文件供代码使用
    if not prepare_table_data_from_string(table_str):
        return {
            'parsed_prediction': 'False',
            'Parse@1': True,
            'ecr_1': False,
            'similarity_score': 0.0,
            'exact_match': False,
            'error': 'Failed to prepare table data'
        }
    
    # 将代码放入主函数中以便执行
    main_code = "if __name__ == '__main__':\n"
    for line in python_code.strip().split('\n'):
        main_code += f"    {line}\n"
    
    # 尝试执行代码
    ecr_1 = False
    try:
        # 记录原始标准输出
        old_stdout = sys.stdout
        sys.stdout = StringIO()
        
        # 执行代码
        execute_with_timeout(main_code)
        ecr_1 = True
        
        # 恢复标准输出
        sys.stdout = old_stdout
    except Exception as e:
        sys.stdout = old_stdout
        error_msg = str(e)
        print(f"代码执行错误: {error_msg}")
        return {
            'parsed_prediction': 'False',
            'Parse@1': True,
            'ecr_1': False,
            'similarity_score': 0.0,
            'exact_match': False,
            'error': error_msg
        }
    
    # 获取图表数据
    prediction_data = None
    exact_match = False
    
    try:
        # 基于图表类型提取数据
        if chart_type == 'line':
            prediction_data = get_line_y_predictions(plt)
        elif chart_type == 'bar':
            prediction_data = get_bar_y_predictions(plt)
        elif chart_type == 'hbar':
            prediction_data = get_hbar_y_predictions(plt)
        elif chart_type == 'pie':
            prediction_data = get_pie_y_predictions(plt)
        elif chart_type == 'area':
            prediction_data = get_area_y_predictions(plt)
        elif chart_type == 'radar':
            prediction_data = get_radar_y_predictions(plt)
        elif chart_type == 'scatter':
            prediction_data = get_scatter_y_predictions(plt)
        elif chart_type == 'waterfall':
            prediction_data = get_waterfall_y_predictions(plt)
            
        # 使用两种评估方法
        # 1. 严格的排序匹配评估
        if chart_type == 'pie':
            exact_match = compute_pie_chart_metric(reference_data, prediction_data)
        else:
            exact_match = compute_general_chart_metric(reference_data, prediction_data)
            
        # 2. 相似度评估
        similarity_score = compute_chart_similarity(reference_data, prediction_data)
        
        # 清除图表以避免影响下一个评估
        plt.close('all')
        
        # 如果相似度超过阈值，认为预测正确
        is_similar = similarity_score > 0.7
        
        return {
            'parsed_prediction': str(is_similar or exact_match),  # 如果任一标准通过则标记为正确
            'Parse@1': True,
            'ecr_1': ecr_1,
            'similarity_score': similarity_score,
            'exact_match': exact_match
        }
    except Exception as e:
        plt.close('all')
        print(f"图表数据提取错误: {str(e)}")
        return {
            'parsed_prediction': 'False',
            'Parse@1': True,
            'ecr_1': ecr_1,
            'similarity_score': 0.0,
            'exact_match': False,
            'error': str(e)
        }

def load_reference_data(reference_jsonl_path):
    """从JSONL文件中加载参考数据"""
    reference_data = {}
    try:
        with open(reference_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if 'id' in data:
                            reference_data[data['id']] = data
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"加载参考数据错误: {str(e)}")
    return reference_data

# def load_answers(file_path):
#     """加载LLM回答文件"""
#     try:
#         with open(file_path, 'r', encoding='utf-8') as f:
#             return json.load(f)
#     except (FileNotFoundError, json.JSONDecodeError) as e:
#         print(f"错误: 无法加载答案文件 {file_path}: {str(e)}")
#         return {}
def load_answers(file_path):
    """加载LLM回答文件，处理列表或字典格式"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # 如果加载的是列表，转换为以id为键的字典
            if isinstance(data, list):
                answers_dict = {}
                for item in data:
                    if 'id' in item:
                        answers_dict[item['id']] = item
                    # 如果没有id字段，尝试使用索引作为键
                    else:
                        answers_dict[str(len(answers_dict))] = item
                return answers_dict
            # 如果已经是字典，直接返回
            elif isinstance(data, dict):
                return data
            else:
                print(f"错误: 无法解析答案文件 {file_path}，格式不是列表或字典")
                return {}
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"错误: 无法加载答案文件 {file_path}: {str(e)}")
        return {}

def extract_final_answer(answer_text):
    """提取'Final Answer:'后面的内容作为预测答案"""
    if not answer_text:
        return ""
    pattern = r"Final Answer:\s*(.*?)\s*$"
    match = re.search(pattern, answer_text, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(1).strip()
    else:
        return answer_text.strip()

def calculate_rouge_l(prediction, reference):
    """计算单个问题的ROUGE-L分数"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    
    pred_str = str(prediction) if prediction is not None else ""
    ref_str = str(reference) if reference is not None else ""
    
    score = scorer.score(ref_str, pred_str)
    return score['rougeL'].fmeasure

def extract_causal_relation(text):
    """从预测或参考文本中提取因果关系描述"""
    # 尝试提取正向、负向或无明显影响的描述
    positive_pattern = r"positive\s+impact|positively|increases|increase|elevated|elevates|higher|rise|rises|raising|raised|growth|growing|improve|improves|enhancement|enhances|boost|boosts"
    negative_pattern = r"negative\s+impact|negatively|decreases|decrease|reduced|reduces|inverse|lower|decline|declines|declining|fall|falls|falling|dropped|dropping|deteriorate|deteriorates|weakens|weaken"
    neutral_pattern = r"no\s+(?:clear|significant|causal|obvious)?\s*(?:impact|influence|correlation|relationship|link|effect)|negligible|doesn't\s+(?:causally)?\s+influence|not\s+(?:causally)?\s+influence|no\s+clear\s+correlation|no\s+correlation|none\s+of\s+them"
    
    if re.search(positive_pattern, text, re.IGNORECASE):
        return "Positive"
    elif re.search(negative_pattern, text, re.IGNORECASE):
        return "Negative" 
    elif re.search(neutral_pattern, text, re.IGNORECASE):
        return "Neutral"
    return "Unknown"

def evaluate_causal_analysis(pred_text, ref_text):
    """评估因果分析问题的正确性"""
    pred_relation = extract_causal_relation(pred_text)
    ref_relation = extract_causal_relation(ref_text)
    
    is_correct = pred_relation == ref_relation
    
    return {
        'relation_match': is_correct,
        'pred_relation': pred_relation,
        'ref_relation': ref_relation,
        'is_correct': is_correct
    }


def evaluate_by_task_type(answers_file, reference_jsonl_file, output_file):
    """根据任务类型进行分层评估，只按主类型分组，非可视化任务统一使用ROUGE-L评估"""
    # 加载数据
    answers = load_answers(answers_file)
    reference_data = load_reference_data(reference_jsonl_file)
    
    print(f"加载了 {len(answers)} 个回答和 {len(reference_data)} 个参考数据")
    
    # 只按主类型分组
    task_types = defaultdict(list)
    
    # 处理所有问题
    all_questions = []
    
    # 用于统计平均分数
    all_scores = []
    
    # 处理answers (已经转换为字典格式)
    for qid, qdata in answers.items():
        if qid not in reference_data:
            continue
            
        ref_data = reference_data[qid]
        
        # 获取问题类型 - 只使用主类型
        qtype = ref_data.get('qtype', '')
        qsubtype = ref_data.get('qsubtype', '')
        
        if not qtype:
            continue
            
        # 提取预测答案 - 如果model_answer不存在，尝试直接使用answer字段
        model_answer = qdata.get('model_answer', qdata.get('answer', ''))
        pred = extract_final_answer(model_answer)
        
        # 获取真实答案
        ref = ref_data.get('answer', '')
        
        # 准备问题数据
        question_data = {
            'id': qid,
            'prediction': pred,
            'reference': ref,
            'qtype': qtype,
            'qsubtype': qsubtype,  # 仍然记录子类型，但不用于分组和评估
            'instruction': ref_data.get('instruction', '')[:200] + '...',
            'evaluation': {}  # 评估结果将存放在这里
        }
        
        # 根据任务类型执行不同的评估
        if qtype == 'Visualization':
            try:
                # 获取表格数据和图表类型
                table_str = ref_data.get('table', None)
                
                # 如果表格数据不在ref_data中，尝试从指令中提取
                if not table_str and 'instruction' in ref_data:
                    table_str = extract_table_from_instruction(ref_data['instruction'])
                
                # 如果仍然没有表格数据，使用空表格
                if not table_str:
                    table_str = '{"columns":[], "data":[]}'
                    
                chart_type = ref_data.get('chart_type', '') 
                
                # 如未指定图表类型，尝试推断
                if not chart_type:
                    question = ref_data.get('instruction', '')
                    if re.search(r'line|trend|time series', question, re.IGNORECASE):
                        chart_type = 'line'
                    elif re.search(r'horizontal bar|hbar', question, re.IGNORECASE):
                        chart_type = 'hbar'
                    elif re.search(r'pie|proportion|percentage', question, re.IGNORECASE):
                        chart_type = 'pie'
                    elif re.search(r'scatter|correlation|point', question, re.IGNORECASE):
                        chart_type = 'scatter'
                    elif re.search(r'radar|spider|web', question, re.IGNORECASE):
                        chart_type = 'radar'
                    elif re.search(r'area|cumulative', question, re.IGNORECASE):
                        chart_type = 'area'
                    else:
                        chart_type = 'bar'  # 默认为条形图
                
                # 解析参考答案
                reference_str = ref.replace('y_references = ', '')
                try:
                    reference_result = eval(reference_str)
                except:
                    print(f"无法解析问题 {qid} 的参考答案: {reference_str}")
                    reference_result = []
                
                # 调用可视化评估函数
                viz_result = evaluate_chart_code(qdata.get('model_answer', ''), table_str, reference_result, chart_type)
                question_data['evaluation'] = viz_result
                question_data['chart_type'] = chart_type

                # 记录分数 - 使用相似度作为统一分数
                score = viz_result.get('similarity_score', 0.0)
                question_data['score'] = score
                all_scores.append(score)
                
                # 添加布尔字段标记是否正确 (相似度>0.7 或 完全匹配)
                is_correct = viz_result.get('parsed_prediction') == 'True'
                question_data['evaluation']['is_correct'] = is_correct
        
                
            except Exception as e:
                print(f"可视化评估异常 ({qid}): {str(e)}")
                question_data['evaluation'] = {
                    'parsed_prediction': 'False',
                    'Parse@1': False,
                    'ecr_1': False,
                    'similarity_score': 0.0,
                    'exact_match': False,
                    'error': f"Evaluation error: {str(e)}"
                }
                question_data['score'] = 0.0
                all_scores.append(0.0)
                question_data['evaluation']['is_correct'] = False
        
        else:
            # 所有非可视化任务统一使用ROUGE-L评估
            rouge_l = calculate_rouge_l(pred, ref)
            # exact_match = pred.lower().strip() == ref.lower().strip()
            is_correct = rouge_l > 0.5  # 使用统一的阈值判断
            
            question_data['evaluation'] = {
                'rouge_l': rouge_l,
                # 'exact_match': exact_match,
                'is_correct': is_correct
            }
            
            # 记录分数 - 使用ROUGE-L作为分数
            question_data['score'] = rouge_l
            all_scores.append(rouge_l)
        
        # 将问题添加到对应类型的列表和全部问题列表 - 仅使用主类型(qtype)
        task_types[qtype].append(question_data)
        all_questions.append(question_data)
    
    # 计算每种类型的评估指标
    results = {
        'overall': {
            'total': len(all_questions),
            'accuracy': 0,
            'average_score': 0,
            'details': {}
        },
        'by_type': {}
    }
    
    # 处理按类型统计
    for task_type, questions in task_types.items():
        type_stats = {
            'total': len(questions),
            'correct': 0,
            'average_score': 0,
            'metrics': {}
        }
        
        # 收集分数
        type_scores = [q.get('score', 0.0) for q in questions]
        type_stats['average_score'] = sum(type_scores) / len(type_scores) if type_scores else 0
        
        # 可视化任务特殊处理
        if task_type == 'Visualization':
            parsed = 0
            executed = 0
            correct = 0
            exact_match = 0
            similarity_sum = 0
            
            for q in questions:
                eval_result = q['evaluation']
                if eval_result.get('Parse@1', False):
                    parsed += 1
                if eval_result.get('ecr_1', False):
                    executed += 1
                if eval_result.get('parsed_prediction') == 'True':
                    correct += 1
                    type_stats['correct'] += 1
                if eval_result.get('exact_match', False):
                    exact_match += 1
                similarity_sum += eval_result.get('similarity_score', 0.0)
            
            # 计算各项指标
            type_stats['metrics'] = {
                'parse_rate': parsed / len(questions) if len(questions) > 0 else 0,
                'execution_rate': executed / len(questions) if len(questions) > 0 else 0,
                'accuracy': correct / len(questions) if len(questions) > 0 else 0,
                # 'exact_match_rate': exact_match / len(questions) if len(questions) > 0 else 0,
                'avg_similarity': similarity_sum / len(questions) if len(questions) > 0 else 0
            }
        
        # 所有非可视化任务统一评估处理
        else:
            # 计算各种统计指标
            correct = 0
            rouge_sum = 0
            # exact_matches = 0
            
            for q in questions:
                eval_result = q['evaluation']
                
                # 计算正确题目数
                if eval_result.get('is_correct', False):
                    correct += 1
                    type_stats['correct'] += 1
                
                # 累加ROUGE-L分数
                rouge_sum += eval_result.get('rouge_l', 0)
                
                # 统计完全匹配数
                if eval_result.get('exact_match', False):
                    exact_matches += 1
            
            # 基本指标
            type_stats['metrics'] = {
                'accuracy': correct / len(questions) if len(questions) > 0 else 0,
                'avg_rouge_l': rouge_sum / len(questions) if len(questions) > 0 else 0,
                # 'exact_match_rate': exact_matches / len(questions) if len(questions) > 0 else 0
            }
        
        results['by_type'][task_type] = type_stats
    
    # 计算总体准确率和平均分数
    total_correct = sum(stats['correct'] for stats in results['by_type'].values())
    total_questions = sum(stats['total'] for stats in results['by_type'].values())
    results['overall']['accuracy'] = total_correct / total_questions if total_questions > 0 else 0
    results['overall']['average_score'] = sum(all_scores) / len(all_scores) if all_scores else 0
    
    # 添加详细的每个问题评估结果
    results['overall']['details'] = sorted(all_questions, key=lambda x: x['id'])
    
    # 保存结果
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # 打印评估摘要
    print(f"评估完成，结果已保存至 {output_file}")
    print(f"总体准确率: {results['overall']['accuracy']:.4f} ({total_correct}/{total_questions})")
    print(f"总体平均分数: {results['overall']['average_score']:.4f}")
    print("\n按任务类型的评估结果:")
    
    for task_type, stats in results['by_type'].items():
        print(f"\n** {task_type} ({stats['total']}题) **")
        print(f"  准确率: {stats['metrics'].get('accuracy', 0):.4f} ({stats['correct']}/{stats['total']})")
        print(f"  平均分数: {stats['average_score']:.4f}")
        
        # 针对不同类型显示特定指标
        if task_type == 'Visualization':
            print(f"  解析成功率: {stats['metrics'].get('parse_rate', 0):.2%}")
            print(f"  执行成功率: {stats['metrics'].get('execution_rate', 0):.2%}")
            print(f"  图表正确率: {stats['metrics'].get('accuracy', 0):.2%}")
            # print(f"  完全匹配率: {stats['metrics'].get('exact_match_rate', 0):.2%}")
            print(f"  平均相似度: {stats['metrics'].get('avg_similarity', 0):.4f}")
        else:
            # 所有非可视化任务统一显示ROUGE-L和精确匹配率
            print(f"  平均ROUGE-L: {stats['metrics'].get('avg_rouge_l', 0):.4f}")
            # print(f"  精确匹配率: {stats['metrics'].get('exact_match_rate', 0):.4f}")

    return results

def main():
    parser = argparse.ArgumentParser(description='评估LLM在各种表格任务上的表现')
    parser.add_argument('--input', type=str, default='/netcache/mengjinxiang/Project/LLaMA-Factory-main/results/tablebench_results.json',
                        help='包含LLM回答的JSON文件路径')
    parser.add_argument('--reference', type=str, default='/netcache/mengjinxiang/Project/LLaMA-Factory-main/data/tablebench/TableBench_TCoT_new.jsonl',
                        help='参考答案的JSONL文件路径')
    parser.add_argument('--output', type=str, default='evaluation_results.json',
                        help='评估结果输出文件路径')
    
    args = parser.parse_args()
    evaluate_by_task_type(args.input, args.reference, args.output)

if __name__ == '__main__':
    main()