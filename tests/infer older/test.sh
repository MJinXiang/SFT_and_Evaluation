# ===================== 参数读取 ============================
MODEL_PATH="/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"   # 模型路径，比如 /mnt/usercache/huggingface/Qwen2.5-3B-Instruct
TASK_NAME="tatqa"  # 任务名，比如 tatqa
TRAIN_TYPE="no"  # 训练方式，比如 grpo、ppo、sft
MODEL_SIZE="3b"    # 模型大小，比如 3b、7b
TENSOR_PARALLEL_SIZE=2  # 张量并行大小，为了与注意力头数量匹配
BATCH_SIZE=128      # 批处理大小

# ===================== 路径配置 ============================
BASE_PATH="$(pwd)"  # 使用当前目录作为基础路径，如：/mnt/usercache/mengjinxiang/Project/SFT_and_Evaluation
INFER_SCRIPT="${BASE_PATH}/tests/${TASK_NAME}.py"
EVAL_SCRIPT="${BASE_PATH}/tests/eval/${TASK_NAME}_eval.py"

# 自动生成输出文件路径
PRED_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}.json"
INFER_LOG="${BASE_PATH}/results/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_infer.log"
EVAL_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_eval_results.json"

# 确保输出目录存在
mkdir -p "$(dirname "$PRED_OUTPUT")"
mkdir -p "$(dirname "$INFER_LOG")"

# ===================== 运行推理 ============================
echo "Running inference with VLLM..."
python "$INFER_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --output_file "$PRED_OUTPUT" \
    --log_file "$INFER_LOG" \
    --base_path "$BASE_PATH" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --batch_size $BATCH_SIZE \
    --temperature 0.0

if [ $? -ne 0 ]; then
  echo "Inference failed"
  exit 1
fi

# ===================== 运行评估 ============================
echo "Running evaluation..."
python "$EVAL_SCRIPT" \
    --results_file "$PRED_OUTPUT" \
    --output_file "$EVAL_OUTPUT" \
    --base_path "$BASE_PATH" 

if [ $? -ne 0 ]; then
  echo "Evaluation failed"
  exit 1
fi

echo "Testing and evaluation completed successfully!"




# # ===================== 参数读取 ============================
# MODEL_PATH="/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"   # 模型路径，比如 /mnt/usercache/huggingface/Qwen2.5-3B-Instruct
# TASK_NAME="tatqa"  # 任务名，比如 tatqa
# TRAIN_TYPE="grpo"  # 训练方式，比如 grpo、ppo、sft
# MODEL_SIZE="3b"    # 模型大小，比如 3b、7b
# API_PORT=8000      # 端口号，比如 8000

# # ===================== 路径配置 ============================
# BASE_PATH="$(pwd)"  # 使用当前目录作为基础路径，如：/mnt/usercache/mengjinxiang/Project/LLaMA-Factory-main
# CONFIG_FILE="examples/inference/${TASK_NAME}.yaml"
# API_LOG_FILE="${BASE_PATH}/logs/${TASK_NAME}_${TRAIN_TYPE}_api.log"
# INFER_SCRIPT="${BASE_PATH}/tests/${TASK_NAME}.py"
# EVAL_SCRIPT="${BASE_PATH}/tests/eval/${TASK_NAME}_eval.py"

# # 自动生成输出文件路径
# PRED_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}.json"
# INFER_LOG="${BASE_PATH}/results/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_infer.log"
# EVAL_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_eval_results.json"

# # === 动态替换 config 文件中的模型路径 ===
# echo "Updating model path in config file: $CONFIG_FILE"
# sed -i "s|^model_name_or_path:.*|model_name_or_path: ${MODEL_PATH}|" "$CONFIG_FILE"

# # ===================== 启动 API ============================
# echo "Starting inference API on port $API_PORT..."
# API_PORT=$API_PORT llamafactory-cli api "$CONFIG_FILE" > "$API_LOG_FILE" 2>&1 &
# API_PID=$!
# echo "API started (PID: $API_PID)"

# # 等待初始化
# echo "Waiting for API initialization..."
# sleep 30

# # ===================== 运行推理 ============================
# echo "Running inference test..."
# python "$INFER_SCRIPT" \
#     --api_port $API_PORT \
#     --output_file "$PRED_OUTPUT" \
#     --model_path "$MODEL_PATH" \
#     --log_file "$INFER_LOG" \
#     --base_path "$BASE_PATH" \
#     --temperature 0.6

# if [ $? -ne 0 ]; then
#   echo "Inference failed, stopping API..."
#   kill $API_PID
#   exit 1
# fi

# # ===================== 关闭 API ============================
# echo "Stopping API server..."
# kill $API_PID

# # ===================== 运行评估 ============================
# echo "Running evaluation..."
# python "$EVAL_SCRIPT" \
#     --results_file "$PRED_OUTPUT" \
#     --output_file "$EVAL_OUTPUT" \
#     --base_path "$BASE_PATH" 

# if [ $? -ne 0 ]; then
#   echo "Evaluation failed"
#   exit 1
# fi

# echo "Testing and evaluation completed successfully!"
