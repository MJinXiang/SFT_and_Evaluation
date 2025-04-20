import json
import re
import os
import sys
import argparse
import random
import numpy as np
from collections import defaultdict

def extract_sql(solution_str):
    """从模型回答中提取SQL查询，支持从标准格式和标签封装格式中提取"""
    if solution_str is None:
        return None
    
    # 首先检查是否有<answer>标签，提取其中内容
    answer_tag_pattern = r'<answer>(.*?)</answer>'
    answer_tag_match = re.search(answer_tag_pattern, solution_str, re.DOTALL)
    if answer_tag_match:
        # 如果找到<answer>标签，只处理标签内的内容
        solution_str = answer_tag_match.group(1).strip()
    
    # 现在处理提取的内容或原始内容
    
    # 尝试从Answer标记后提取SQL
    answer_pattern = r'Answer:\s*```sql\s*(.*?)\s*(?:```|$)'
    answer_matches = re.findall(answer_pattern, solution_str, re.DOTALL | re.IGNORECASE)
    if answer_matches:
        return answer_matches[-1].strip()
    
    # 如果没有明确的Answer标记，尝试从Markdown代码块中提取
    md_pattern = r'```sql\s*(.*?)\s*```'
    md_matches = re.findall(md_pattern, solution_str, re.DOTALL)
    if md_matches:
        return md_matches[-1].strip()
    
    # 尝试提取顶层代码块
    code_pattern = r'```\s*(.*?)\s*```'
    code_matches = re.findall(code_pattern, solution_str, re.DOTALL)
    if code_matches:
        return code_matches[-1].strip()
    
    # 检查是否包含SELECT字样
    if "SELECT" in solution_str.upper():
        # 尝试提取完整SQL语句
        sql_line_pattern = r'SELECT\s+.*?FROM\s+.*?(WHERE\s+.*?)?(?:;|$)'
        sql_line_matches = re.findall(sql_line_pattern, solution_str, re.IGNORECASE | re.DOTALL)
        if sql_line_matches:
            return sql_line_matches[-1].strip()
    
    return None


def normalize_sql(sql_string):
    """标准化SQL查询字符串，使其更宽松地匹配"""
    if sql_string is None:
        return ""
    
    # 基本清理
    sql_string = ' '.join(sql_string.split()).lower()
    sql_string = sql_string.replace("'", "").replace('"', "")  # 去除引号
    
    # 处理分号
    sql_string = sql_string.rstrip(';')
    
    # 标准化列名 - 将下划线替换为空格
    sql_string = sql_string.replace("_", " ")
    
    # 标准化聚合函数
    agg_funcs = ['count', 'sum', 'avg', 'min', 'max']
    for func in agg_funcs:
        pattern = r'{}[ ]*\(([^)]+)\)'.format(func)
        replacement = r'{} \1'.format(func)
        sql_string = re.sub(pattern, replacement, sql_string)
    
    return sql_string



def parse_sql(query, columns, verbose=False):
    """解析SQL查询并提取结构化信息，增强对包含空格列名的支持和多条件处理"""
    if query is None:
        return {
            "agg": 0,
            "conds": {"column_index": [], "condition": [], "operator_index": []},
            "human_readable": "",
            "sel": -1
        }
    
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']

    # 创建列名的副本，避免修改原始列表
    columns = list(columns)
    
    # 创建列名映射字典，用于模糊匹配
    col_map = {}
    col_map_lower = {}  # 用于不区分大小写的匹配
    for i, col in enumerate(columns):
        # 标准化列名（小写、移除空格和括号）
        norm_col = col.lower().strip().replace('(', '').replace(')', '').replace(' ', '')
        col_map[norm_col] = i
        col_map_lower[col.lower().strip()] = i  # 仅小写转换，保留空格
    
    # 提取SELECT目标列
    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', query, re.IGNORECASE | re.DOTALL)
    select_clause = select_match.group(1).strip() if select_match else None
    
    if select_clause is None:
        return {
            "agg": 0,
            "conds": {"column_index": [], "condition": [], "operator_index": []},
            "human_readable": query,
            "sel": -1
        }
    
    # 处理聚合函数
    agg = 0
    select_col = select_clause
    for i, op in enumerate(agg_ops[1:], 1):
        pattern = rf'{op}\s*\((.*?)\)'
        match = re.search(pattern, select_clause, re.IGNORECASE)
        if match:
            agg = i
            select_col = match.group(1).strip()
            break
    
    # 尝试匹配列名
    sel = -1
    # 1. 直接匹配
    if select_col in columns:
        sel = columns.index(select_col)
    # 2. 不区分大小写匹配
    elif select_col.lower() in col_map_lower:
        sel = col_map_lower[select_col.lower()]
    # 3. 标准化匹配
    else:
        select_col_norm = select_col.lower().strip().replace('(', '').replace(')', '').replace(' ', '')
        if select_col_norm in col_map:
            sel = col_map[select_col_norm]
    
    # 提取WHERE条件
    conds = {"column_index": [], "condition": [], "operator_index": []}
    where_match = re.search(r'WHERE\s+(.*?)(?:;|$)', query, re.IGNORECASE | re.DOTALL)
    
    if where_match:
        conditions_str = where_match.group(1).strip()
        # 处理AND或OR连接的条件 - 更好地处理连接符号
        conditions = re.split(r'\s+AND\s+|\s+OR\s+', conditions_str, flags=re.IGNORECASE)
        
        for cond in conditions:
            # 移除前后空格
            cond = cond.strip()
            if not cond:
                continue
                
            # 针对每个条件，尝试提取列名、操作符和值
            # 先尝试各种操作符
            matched = False
            for op_idx, op in enumerate(cond_ops):
                # 使用正则表达式精确匹配操作符
                pattern = r'(.+?)\s*' + re.escape(op) + r'\s*(.+)'
                match = re.search(pattern, cond)
                
                if match:
                    matched = True
                    col_name = match.group(1).strip()
                    value = match.group(2).strip().strip("'").strip('"')
                    
                    # 尝试匹配列名
                    col_idx = -1
                    
                    # 尝试直接匹配
                    if col_name in columns:
                        col_idx = columns.index(col_name)
                    # 尝试不区分大小写匹配
                    elif col_name.lower() in col_map_lower:
                        col_idx = col_map_lower[col_name.lower()]
                    # 尝试标准化匹配
                    else:
                        # 去除空格和符号比较
                        col_norm = col_name.lower().replace(' ', '').replace('_', '')
                        for i, col in enumerate(columns):
                            col_clean = col.lower().replace(' ', '').replace('_', '')
                            if col_clean == col_norm:
                                col_idx = i
                                break
                    
                        # 子字符串匹配
                        if col_idx < 0:
                            for i, col in enumerate(columns):
                                if col.lower() in col_name.lower() or col_name.lower() in col.lower():
                                    col_idx = i
                                    break
                    
                    # 如果找到了列索引，添加到条件中
                    if col_idx >= 0:
                        conds["column_index"].append(col_idx)
                        conds["condition"].append(value)
                        conds["operator_index"].append(op_idx)
                    elif verbose:  # 只在详细模式下显示警告
                        print(f"警告: 无法匹配列名 '{col_name}' 到表头列表: {columns}")
                    
                    break  # 找到操作符后跳出循环
    
    return {
        "agg": agg,
        "conds": conds,
        "human_readable": query,
        "sel": sel
    }

def score_sql(parsed_sql, correct_sql):
    """
    比较预测的SQL与正确的SQL的结构化信息
    只有所有组件完全匹配才返回1分，否则返回0分
    """
    # 检查选择的列是否匹配
    if parsed_sql["sel"] != correct_sql["sel"]:
        return 0
    
    # 检查聚合函数是否匹配
    if parsed_sql["agg"] != correct_sql["agg"]:
        return 0
    
    parsed_conds = parsed_sql["conds"]
    correct_conds = correct_sql["conds"]
    
    # 检查条件列索引、条件值和操作符是否完全匹配（无视顺序）
    if sorted(parsed_conds["column_index"]) != sorted(correct_conds["column_index"]):
        return 0
    
    # 对条件列表排序并逐一比较
    sorted_parsed = sorted(zip(parsed_conds["column_index"], parsed_conds["condition"], parsed_conds["operator_index"]))
    sorted_correct = sorted(zip(correct_conds["column_index"], correct_conds["condition"], correct_conds["operator_index"]))
    
    for i in range(len(sorted_parsed)):
        # 检查列索引
        if sorted_parsed[i][0] != sorted_correct[i][0]:
            return 0
            
        # 检查操作符
        if sorted_parsed[i][2] != sorted_correct[i][2]:
            return 0
            
        # 比较条件值，更宽松地进行匹配
        parsed_value = str(sorted_parsed[i][1]).lower().strip()
        correct_value = str(sorted_correct[i][1]).lower().strip()
        
        # 标准化条件值，将下划线替换为空格
        parsed_value = parsed_value.replace("_", " ")
        correct_value = correct_value.replace("_", " ")
        
        # 移除可能的引号和括号
        parsed_value = parsed_value.replace("'", "").replace('"', "").replace("(", "").replace(")", "")
        correct_value = correct_value.replace("'", "").replace('"', "").replace("(", "").replace(")", "")
        
        if parsed_value != correct_value:
            return 0
    
    # 所有组件都匹配，返回1分
    return 1.0
        

def compute_score(solution_str, ground_truth, table, ans):
    """
    计算WikiSQL评分：
    1. 如果SQL语句完全相同，得1分
    2. 如果SQL语句不同，但所有结构化组件都完全匹配，得1分
    3. 否则得0分
    
    Args:
        solution_str: 模型生成的回答
        ground_truth: 真实SQL查询字符串
        table: 表结构字典，包含header字段
        ans: 真实答案的结构化表示
    
    Returns:
        评分 (0或1)
    """
    # 提取预测的SQL
    predicted_sql = extract_sql(solution_str=solution_str)
    
    # 随机打印调试信息
    do_print = random.randint(1, 64) == 1
    
    if do_print:
        print(f"--------------------------------")
        print(f"Ground Truth SQL: {ground_truth}")
        print(f"Generation SQL: {predicted_sql}")
        print(f"Solution string: {solution_str}")
    
    if predicted_sql is None:
        if do_print:
            print(f"No SQL query found")
        return 0
    
    # 检查SQL文本完全匹配
    predicted_sql_normalize = normalize_sql(predicted_sql)
    ground_truth_sql_normalize = normalize_sql(ground_truth)
    
    if predicted_sql_normalize == ground_truth_sql_normalize:
        if do_print:
            print(f"Correct SQL query: PreSQL: {predicted_sql}, GoldSQL: {ground_truth}")
        return 1.0
    
    # SQL文本不匹配，检查结构化信息
    predicted_answer = parse_sql(predicted_sql, table['header'], verbose=do_print)
    final_score = score_sql(predicted_answer, ans)
    
    if do_print:
        if final_score == 1.0:
            print(f"Structurally correct: Components match exactly")
        else:
            print(f"Incorrect: Components do not match")
        print(f"Predicted: {predicted_answer}")
        print(f"Expected: {ans}")
    
    return final_score

def load_test_data_headers(test_data_file):
    """
    从测试数据文件中加载所有问题及其对应的表头信息
    返回一个字典，键为问题文本，值为表头列表
    """
    headers_by_question = {}
    try:
        # 尝试逐行读取JSON对象
        with open(test_data_file, 'r', encoding='utf-8') as f:
            line_num = 0
            for line in f:
                line_num += 1
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                    
                try:
                    # 解析单行JSON
                    item = json.loads(line)
                    question = item.get('question', '')
                    if 'table' in item and 'header' in item['table']:
                        headers_by_question[question] = item['table']['header']
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析错误，已跳过: {str(e)}")
        
        print(f"成功从测试数据文件中加载了{len(headers_by_question)}个问题的表头")
    except Exception as e:
        print(f"加载测试数据文件时出错: {str(e)}")
    
    return headers_by_question



def evaluate_single_item(item, default_header=None):
    """评估单个JSON项目"""
    # 提取所需信息
    model_answer = item.get('model_answer', '')
    truth_sql = item.get('truth_sql', '')
    truth_answer = item.get('truth_answer', {})
    question = item.get('question', '')  # 获取问题，用于推断列名
    
    # 使用提供的表头，如果没有提供则使用一个默认通用表头
    if default_header:
        header_columns = default_header
    else:
        # 使用一个非常通用的默认表头
        header_columns = ["column_0", "column_1", "column_2", "column_3", "column_4", 
                         "column_5", "column_6", "column_7", "column_8", "column_9"]
    
    # 确保表头长度至少能覆盖所需的列索引
    if truth_answer.get('sel', -1) >= len(header_columns):
        header_columns.extend(['column_' + str(i) for i in range(len(header_columns), truth_answer['sel'] + 1)])
    
    for idx in truth_answer.get('conds', {}).get('column_index', []):
        if idx >= len(header_columns):
            header_columns.extend(['column_' + str(i) for i in range(len(header_columns), idx + 1)])
    
    table = {'header': header_columns}
    
    # 计算得分
    score = compute_score(model_answer, truth_sql, table, truth_answer)
    
    return {
        'id': item.get('id', ''),
        'score': score,
        'prediction': extract_sql(model_answer),
        'ground_truth': truth_sql,
        'header': header_columns,  # 添加表头以便调试
        'question': question  # 添加问题以便调试
    }


def process_json_file(json_file, output_file=None, test_data_file=None):
    """处理包含WikiSQL评估项的JSON文件"""
    try:
        # 加载JSON数据
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 从测试数据文件加载表头信息
        headers_by_question = {}
        if test_data_file:
            headers_by_question = load_test_data_headers(test_data_file)
            if not headers_by_question:
                print(f"警告: 未能从测试数据文件中加载表头信息，将使用默认表头")
        
        # 确定数据格式
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            # 如果是单个项目，放入列表中
            items = [data]
        else:
            print(f"错误: 不支持的JSON格式")
            return
        
        # 评估结果
        results = []
        total_score = 0
        
        # 评估每个项目
        for item in items:
            # 获取该项目的问题
            question = item.get('question', '')
            
            # 尝试获取该问题对应的表头
            item_headers = None
            if question in headers_by_question:
                item_headers = headers_by_question[question]
            
            # 评估该项目
            result = evaluate_single_item(item, item_headers)
            results.append(result)
            total_score += result['score']
        
        # 计算总体准确率
        accuracy = total_score / len(results) if results else 0
        
        # 生成评估报告
        report = {
            'overall_accuracy': accuracy,
            'total_items': len(results),
            'correct_items': total_score,
            'results': results
        }
        
        # 保存结果
        if output_file:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
        
        # 打印结果
        print(f"\n评估结果:")
        print(f"总准确率: {accuracy:.4f} ({int(total_score)}/{len(results)})")
        print(f"评估项目数: {len(results)}")
        if output_file:
            print(f"详细结果已保存至: {output_file}")
        
        return report
        
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(description='Evaluate model answers for WikiSQL questions')
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to JSON file containing predictions')
    parser.add_argument('--output_file', type=str, required=True,
                       help='Path to output file for evaluation results')
    parser.add_argument('--base_path', type=str,
                       help='Base path for the project (optional)')
    parser.add_argument('--test_data', type=str, 
                       default='data/wikisql/wikisql_test.json',
                       help='Path to WikiSQL test data file for accurate header information')

    
    args = parser.parse_args()

    test_path = os.path.join(args.base_path, args.test_data)
    
    # Process with new parameter names
    process_json_file(args.results_file, args.output_file, test_path)

if __name__ == '__main__':
    main()