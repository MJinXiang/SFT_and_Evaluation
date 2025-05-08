import os  
import torch  
from typing import List, Dict, Any, Optional  
from vllm import LLM, SamplingParams  
from transformers import AutoTokenizer  
from datasets import load_dataset
import json


class VLLMGenerator:  
    """  
    A class for generating text using vLLM with support for different models.  
    """  
    
    def __init__(self, args):  
        """  
        Initialize the VLLMGenerator with model and tokenizer.  
        
        Args:  
            args: Configuration arguments containing model information  
        """  
        self.args = args  
        # Default EOS tokens list - can be overridden based on model  
        self.EOS = ["<|im_end|>", "</s>"]  
        
        self.model = LLM(  
            model=args.model,  
            max_model_len=args.max_model_len if hasattr(args, 'max_model_len') else 8192,  
            trust_remote_code=True,  
            distributed_executor_backend='ray',  
            tensor_parallel_size=args.tensor_parallel_size
        )  
        
        self.tokenizer = AutoTokenizer.from_pretrained(args.model)  
        
    def vllm_generation(self, prompts, verbose=True):  

        try:  
            # Apply chat template to prompts  
            chat_prompts = []  
            for prompt in prompts:  
                chat_prompts.append(self.tokenizer.apply_chat_template(  
                    prompt,  
                    tokenize=False,  
                    add_generation_prompt=True,  
                ))  
            
            if verbose:  
                print("Chat prompts:")  
                print(chat_prompts[0])  
            

            # Standard generation parameters  
            vllm_outputs = self.model.generate(  
                prompts=chat_prompts,  
                sampling_params=SamplingParams(  
                    max_tokens=self.args.max_new_tokens,  
                    temperature=self.args.temperature if hasattr(self.args, 'temperature') else 0.0,  
                    top_p=self.args.top_p if hasattr(self.args, 'top_p') else 1.0,  
                    stop=self.EOS,  
                ),  
                use_tqdm=True,  
            )  
            
            # Process generated outputs  
            raw_generations = [x.outputs[0].text for x in vllm_outputs]  
            # gen_strs = [task.postprocess_generation(x, id) for id, x in enumerate(raw_generations)]  
            # generations = [[gen_strs[i]] for i in range(len(gen_strs))]  
            
            # if verbose:  
            #     print("Raw Generations:")  
            #     print(raw_generations[0])  
            #     print("Processed Generations:")  
            #     print(gen_strs[0])  
            
            return raw_generations  
            
        except Exception as e:  
            print(f"Error in vLLM generation: {str(e)}")  
            raise  
    

def main():  
    """  
    Example usage of the VLLMGenerator class.  
    """  
    import argparse  
    
    parser = argparse.ArgumentParser(description="vLLM Generation")  
    parser.add_argument("--model", type=str, required=True, help="Model name or path")  
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output")
    parser.add_argument("--max_new_tokens", type=int, default=8192, help="Maximum number of new tokens to generate")  
    parser.add_argument("--max_model_len", type=int, default=32000, help="Maximum model context length")  
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")  
    parser.add_argument("--top_p", type=float, default=1.0, help="Top-p sampling parameter")  
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--tensor_parallel_size", type=int)
    parser.add_argument("--n_sample",type=int,default=1)
    
    
    
    args = parser.parse_args()  
    mode_name = args.model.split('/')[-1]
    
    # Example task and dataset would be initialized here  
    generator = VLLMGenerator(args)  
    data_path = args.data_path
    dataset = load_dataset("json", data_files=data_path, split="train")
    # dataset = dataset.select(range(512,600))
    instance_ids = dataset["instance_id"]
    output_path = os.path.join(args.output_path, mode_name)
    os.makedirs(output_path,exist_ok=True)
    
    prompts = dataset["messages"]
    for i in range(args.n_sample):  
        save_path = os.path.join(output_path, f"response_temp{args.temperature}_top_p{args.top_p}_sample_{i}.jsonl")
        if os.path.exists(save_path):
            print(f"File {save_path} already exists. Skipping.")
            continue
        generations = generator.vllm_generation(prompts)  
        with open(save_path, 'w') as f:  
            for j, (instance_id, generation) in enumerate(zip(instance_ids, generations)):  
                output = {  
                    "instance_id": instance_id,  # Assuming instance_id exists in dataset  
                    "response": generation  
                }  
                f.write(json.dumps(output) + '\n')  
        
        print(f"Saved {len(generations)} generations to {save_path}")  


    # print(f"Generated {len(generations)} responses")  

if __name__ == "__main__":  
    main()  
    # python infer.py --model /mnt/hdfs/tiktok_aiic/user/liuqian/Qwen2.5-7B --tensor_parallel_size=4 --n_sample 16
    