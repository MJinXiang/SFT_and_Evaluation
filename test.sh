API_PORT=8000  # 如果同时测多个的模型，需要修改这个值
CONFIG_FILE="examples/inference/qwen2.5-3b-tatqa.yaml" # 不用修改这个，但需要修改这个文件里面的模型路径
API_LOG_FILE="llama_api_tatqa.log" # 不用修改
INFER_SCRIPT="./tests/tatqa.py" # 不用修改
EVAL_SCRIPT="./tests/eval/tatqa_eval.py" # 不用修改
MODEL_PATH="/mnt/usercache/huggingface/Qwen2.5-3B-Instruct" # 需要修改
BASE_PATH="/mnt/usercache/mengjinxiang/Project/LLaMA-Factory-main" # 当前项目所在路径

PRED_OUTPUT="./results/tatqa/tatqa_3b_grpo.json" # 文件名需要修改
INFER_LOG="./results/tatqa/tatqa_3b_grpo_infer.log" # 文件名需要修改
EVAL_OUTPUT="./results/tatqa/tatqa_3b_grpo_eval_results.json" # 文件名需要修改

# === Start API server ===
echo "Starting inference API..."
API_PORT=$API_PORT llamafactory-cli api $CONFIG_FILE > $API_LOG_FILE 2>&1 &
API_PID=$!
echo "API started (PID: $API_PID)"

# Wait for API initialization
echo "Waiting for API initialization..."
sleep 30

# === Run inference test ===
echo "Running inference test..."
python $INFER_SCRIPT \
    --api_port $API_PORT \
    --output_file $PRED_OUTPUT \
    --model_path $MODEL_PATH \
    --log_file $INFER_LOG \
    --base_path $BASE_PATH \
    --temperature 0.6

# Check if inference was successful
if [ $? -ne 0 ]; then
  echo "Inference test failed, stopping API..."
  kill $API_PID
  exit 1
fi

# === Stop API server ===
echo "Stopping API server..."
kill $API_PID

# === Run evaluation ===
echo "Running evaluation..."
python $EVAL_SCRIPT \
    --results_file $PRED_OUTPUT \
    --output_file $EVAL_OUTPUT \
    --base_path $BASE_PATH

# Check if evaluation was successful
if [ $? -ne 0 ]; then
  echo "Evaluation failed"
  exit 1
fi

echo "Testing and evaluation completed successfully"