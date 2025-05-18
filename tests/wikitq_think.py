import json
import os
import logging
import time
import sys
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
from prompt import COT_PROMPT_TEMPLATE, CODE_COT_PROMPT_TEMPLATE
from utils.parser import extract_program
import pandas as pd
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Setup logging
def setup_logger(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger('table_qa_processor')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

class VLLMGenerator:  
    def __init__(self, model_path, max_model_len=8192, tensor_parallel_size=1):  
        self.EOS = ["<|im_end|>", "</s>"]  
        self.model = LLM(  
            model=model_path,  
            max_model_len=max_model_len,  
            trust_remote_code=True,  
            distributed_executor_backend='ray',  
            tensor_parallel_size=tensor_parallel_size
        )  
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    def generate(self, prompts, max_new_tokens=2048, temperature=0.0, top_p=1.0, verbose=False):
        try:  
            chat_prompts = []  
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                chat_prompts.append(self.tokenizer.apply_chat_template(  
                    messages, tokenize=False, add_generation_prompt=True))  
            
            # if verbose and chat_prompts:  
            #     print("Example chat prompt:", chat_prompts[0])  
            
            vllm_outputs = self.model.generate(  
                prompts=chat_prompts,  
                sampling_params=SamplingParams(  
                    max_tokens=max_new_tokens,  
                    temperature=temperature,  
                    top_p=top_p,  
                    stop=self.EOS + ["```output"],  
                ),  
                use_tqdm=True,  
            )  
            
            return [x.outputs[0].text for x in vllm_outputs]  
        except Exception as e:  
            print(f"Error in vLLM generation: {str(e)}") 
            print("EOORORORROROROROROROROROR") 
            raise

def create_prompt_from_wikitq_code(item, few_shot_prompts):
    # Clean up headers to remove newlines
    item["table"][0] = [header.replace('\n', ' ') for header in item["table"][0]]
    columns = item["table"][0]
    data_rows = item["table"][1:]

    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    df = pd.DataFrame(data_rows, columns=columns)
    table_str = f"print(df)\n{df}"
    
    return few_shot_prompts + CODE_COT_PROMPT_TEMPLATE.format(table=table_str, question=item["question"])

def get_table_pd(item):
    item["table"][0] = [header.replace('\n', ' ') for header in item["table"][0]]
    columns = item["table"][0]
    data_rows = item["table"][1:]

    pd.set_option('display.expand_frame_repr', False)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)

    return pd.DataFrame(data_rows, columns=columns)

def execute_with_dataframe(code, df):
    # Create namespace with pandas and the dataframe
    namespace = {
        'pd': pd,  # Make pandas available in the code
        'df': df   # Provide the dataframe
    }
    
    try:
        exec(code, namespace)
        
        function_name = next((name for name, obj in namespace.items() 
                           if callable(obj) and name != 'print' and not name.startswith('__')), None)
        
        if function_name:
            result = namespace[function_name](df)
            return result, None
        else:
            return None, "No function found in code"
    except Exception as e:
        return None, f"Error executing code: {str(e)}"

def extract_answer(text):
    """Extract answer between <answer> and </answer> tags"""
    pattern = r'<answer>(.*?)</answer>'
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None

def process_table_qa_data_batch(input_file, output_file, model_path, log_file, max_tokens=2048,
                           temperature=0.0, tensor_parallel_size=1, start_from=0):
    logger = setup_logger(log_file)
    start_time = time.time()
    logger.info(f"Started processing table QA data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Input: {input_file}, Output: {output_file}, Model: {model_path}")
    
    # Initialize VLLM generator
    try:
        generator = VLLMGenerator(
            model_path=model_path,
            max_model_len=16384,
            tensor_parallel_size=tensor_parallel_size
        )
        logger.info(f"VLLM generator initialized successfully")
    except Exception as e:
        logger.error(f"VLLM initialization failed: {e}")
        return
    
    # Read data items
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data_items = [json.loads(line) for line in f if line.strip()]
        logger.info(f"Loaded {len(data_items)} data items")
    except Exception as e:
        logger.error(f"Failed to read input file: {e}")
        return
    
    # Check for intermediate results
    results = []
    processed_ids = set()
    if os.path.exists(f"{output_file}.temp"):
        try:
            with open(f"{output_file}.temp", 'r', encoding='utf-8') as f:
                results = json.load(f)
            processed_ids = {result.get("id", "") for result in results}
            logger.info(f"Loaded {len(results)} intermediate results")
        except Exception as e:
            logger.error(f"Failed to load intermediate results: {e}")
            results = []
            processed_ids = set()
    
    # Filter out already processed items and apply start_from
    remaining_items = [item for idx, item in enumerate(data_items) 
                      if idx >= start_from and item.get("id", f"item_{idx}") not in processed_ids]
    
    logger.info(f"Remaining items to process: {len(remaining_items)}/{len(data_items)}")
    
    success_count = len(results)
    error_count = 0
    
    # Process all data at once
    all_items = [(i, item, create_prompt_from_wikitq_code(item, FEW_SHOT_PROMPTS)) 
                for i, item in enumerate(remaining_items)]
    
    all_prompts = [(i, prompt) for i, _, prompt in all_items]
    end_prompts = []
    
    # Store just the model responses without prompts
    model_responses = {i: "" for i, _ in all_prompts}
    
    max_func_call = 5  # Maximum number of function calls
    max_tokens_allowed = 16384  # Safe buffer
    
    # Run the execution loop
    for epoch in range(max_func_call):
        logger.info(f"Execution epoch {epoch+1}/{max_func_call}")
        current_prompts = all_prompts if epoch == 0 else remain_prompts
        if not current_prompts:
            break
        
        prompts = [item[1] for item in current_prompts]
        prompt_ids = [item[0] for item in current_prompts]
        
        # Check token lengths before sending to model
        prompts_to_process = []
        skipped_indices = []
        
        for idx, (prompt_idx, prompt) in enumerate(zip(prompt_ids, prompts)):
            token_count = len(generator.tokenizer.encode(prompt))
            if token_count  > max_tokens_allowed:
                logger.warning(f"Prompt {prompt_idx} is too long: {token_count} tokens, skipping")
                error_msg = f"\n\nError: Prompt exceeded maximum allowed token length ({token_count} tokens)."
                model_responses[prompt_idx] += error_msg
                skipped_indices.append((idx, prompt_idx, prompt + error_msg))
            else:
                prompts_to_process.append((idx, prompt))
        
        # Add skipped prompts to end_prompts
        for _, prompt_idx, full_prompt in skipped_indices:
            end_prompts.append((prompt_idx, full_prompt))
        
        if not prompts_to_process:
            logger.warning("No prompts to process in this batch after length filtering")
            continue
        
        # Prepare prompts for model
        prompts_for_model = [p for _, p in prompts_to_process]
        prompt_indices = [i for i, _ in prompts_to_process]
        
        try:
            # Generate responses with the model
            logger.info(f"Generating responses for {len(prompts_for_model)} prompts...")
            responses = generator.generate(
                prompts=prompts_for_model,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=1.0
            )
            
            # Reset for next iteration
            remain_prompts = []
            remain_codes = []
            remain_items = []
            
            # Process each response
            for orig_idx, response in zip(prompt_indices, responses):
                i = prompt_ids[orig_idx]
                query = prompts[orig_idx]
                output = response.rstrip()
                model_responses[i] += output
                full_query = query + output
                
                # Check if this output is too long for further processing
                combined_token_count = len(generator.tokenizer.encode(full_query))
                if combined_token_count > max_tokens_allowed - 1000:  # Allow space for one more response
                    logger.warning(f"Response for prompt {i} would be too long for further processing ({combined_token_count} tokens)")
                    model_responses[i] += f"\n\nWarning: Response too long for further processing. Finalizing result."
                    end_prompts.append((i, full_query))
                    continue
                
                if not (output.endswith("</think>") or output.endswith("</answer>") or output.endswith("```") or 
                        output.endswith("</think>\n") or output.endswith("</answer>\n") or output.endswith("```\n")):
                    end_prompts.append((i, full_query))
                elif "<answer>" not in output:
                    program = extract_program(full_query)
                    remain_prompts.append((i, full_query))
                    remain_codes.append(program)
                    item = next((item for idx, item, _ in all_items if idx == i), None)
                    remain_items.append(item)
                else:
                    end_prompts.append((i, full_query))

            # Execute code for prompts that need additional processing
            if remain_codes:
                logger.info(f"Executing {len(remain_codes)} code snippets")
                
                remain_results = []
                for k in range(len(remain_codes)):
                    code = remain_codes[k]
                    item = remain_items[k]
                    
                    try:
                        df = get_table_pd(item)
                        res, report = execute_with_dataframe(code, df)
                        remain_results.append((res, report))
                    except Exception as e:
                        remain_results.append((None, f"Error preparing DataFrame: {str(e)}"))
                
                # Update prompts with execution results
                for k in range(len(remain_prompts)):
                    i, query = remain_prompts[k]
                    res, report = remain_results[k]
                    
                    exec_result = f"\n```output:\n{res}\n```\n" if res is not None else f"\n```Execution error:\n{report}\n```\n"
                    model_responses[i] += exec_result
                    query += exec_result
                    
                    # Check token length again after adding execution results
                    token_count = len(generator.tokenizer.encode(query))
                    if token_count > max_tokens_allowed - 1000:
                        logger.warning(f"Prompt {i} became too long after execution: {token_count} tokens")
                        message = "\nReached maximum token length. Finalizing response."
                        query += message
                        model_responses[i] += message
                        end_prompts.append((i, query))
                        continue

                    if model_responses[i].count("```python") >= 2:
                        message = "\nYou have exceeded the maximum number of Python code usage attempts. You are not allowed to use Python Tool. Do NOT use Python Tool again!!!  Please think step-by-step and give me the final answer directly. Please use output your think process and final answer.\n\n"
                        query += message
                        model_responses[i] += message
                        remain_prompts[k] = (i, query)
                    elif epoch == max_func_call - 1:
                        message = "\nReach max function call limit."
                        query += message
                        model_responses[i] += message
                        end_prompts.append((i, query))
                    else:
                        remain_prompts[k] = (i, query)
                        
        except Exception as e:
            logger.error(f"Error in epoch {epoch+1}: {e}")
            # Only add the affected prompts to end_prompts, not all current prompts
            for i, query in current_prompts:
                if i not in [ep[0] for ep in end_prompts]:  # Only if not already in end_prompts
                    error_msg = f"\n\nError occurred during processing: {str(e)}\n"
                    model_responses[i] += error_msg
                    query += error_msg
                    end_prompts.append((i, query))
            # Don't break - continue with remaining prompts
    
    # Process final results
    end_prompts = sorted(end_prompts, key=lambda x: x[0])
    
    # Record processed items
    for i, (_, item, _) in enumerate(all_items):
        item_start_time = time.time()
        result_idx = next((idx for idx, (j, _) in enumerate(end_prompts) if j == i), -1)
        
        if result_idx != -1:
            full_response = model_responses[i]
            
            # Extract answer from between <answer> and </answer> tags
            extracted_answer = extract_answer(full_response)
            
            item_id = item.get("id", f"item-{i}")
            question = item.get("question", "")
            
            global_item_index = start_from + i + 1
            
            item_time = time.time() - item_start_time
            
            if extracted_answer:
                logger.info(f"Extracted answer: {extracted_answer}")
            logger.info("-" * 50)
            
            result = {
                "id": item_id,
                "source": item.get("source", {}),
                "question": question,
                "answer": item.get("answer", ""),
                "model_answer": full_response,
                "extracted_answer": extracted_answer,
                "processing_time": item_time
            }
            
            results.append(result)
            success_count += 1
        else:
            logger.error(f"Failed to find result for item {i}")
            error_count += 1
    
    # Save results
    with open(f"{output_file}.temp", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results file: {e}")

    # Log summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info(f"Processing completed! Total time: {total_time:.2f} seconds")
    logger.info(f"Successfully processed: {success_count}/{len(data_items)}, Failures: {error_count}")
    if data_items:
        logger.info(f"Success rate: {success_count/len(data_items)*100:.2f}%")
    average_time = total_time / len(remaining_items) if remaining_items else 0
    logger.info(f"Average processing time: {average_time:.2f} seconds per item")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process WikiTableQuestions dataset with VLLM inference')
    
    parser.add_argument('--input_file', type=str, help='Path to input file (defaults to test.jsonl in data/wikitq)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to save results')
    parser.add_argument('--model_path', type=str, required=True, help='Model path or identifier')
    parser.add_argument('--log_file', type=str, required=True, help='Path to log file')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for model generation')
    parser.add_argument('--max_tokens', type=int, default=4096, help='Maximum tokens for model output')
    parser.add_argument('--tensor_parallel_size', type=int, default=2, help='Tensor parallelism size')
    parser.add_argument('--start_from', type=int, default=0, help='Start processing from this index')
    parser.add_argument('--base_path', type=str, help='Base path for the project')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Handle base_path
    base_path = None
    if args.base_path and os.path.exists(args.base_path):
        base_path = args.base_path
    else:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(current_dir) == "tests":
            base_path = os.path.dirname(current_dir)
    
    if not base_path:
        print("Error: Unable to find project root directory")
        exit(1)
    
    print(f"Using root path: {base_path}")
    
    # Set file paths
    input_file = args.input_file if args.input_file else os.path.join(base_path, "data/wikitq/test.jsonl")

    # Load few-shot prompts
    global FEW_SHOT_PROMPTS
    try:
        with open(os.path.join(base_path, "/cpfs01/data/shared/Group-m6/fangyu.lfy/lfy/SFT_and_Evaluation/tests/few_shots/table_r1_think.md"), 'r', encoding='utf-8') as f:
            FEW_SHOT_PROMPTS = f.read()
    except Exception as e:
        print(f"Warning: Could not load few-shot prompts: {e}")
        FEW_SHOT_PROMPTS = ""
    
    # Ensure output and log directories exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    # Process data
    process_table_qa_data_batch(
        input_file=input_file, 
        output_file=args.output_file, 
        model_path=args.model_path, 
        log_file=args.log_file, 
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        start_from=args.start_from
    )

if __name__ == "__main__":
    main()