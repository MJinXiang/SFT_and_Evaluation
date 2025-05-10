# # ===================== 参数读取 ============================
# MODEL_PATH="/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"   # 模型路径，比如 /mnt/usercache/huggingface/Qwen2.5-3B-Instruct
# TASK_NAME="ottqa"  # 任务名，比如 tatqa
# TRAIN_TYPE="base"  # 训练方式，比如 grpo、ppo、sft
# MODEL_SIZE="3b"    # 模型大小，比如 3b、7b
# TENSOR_PARALLEL_SIZE=2  # 张量并行大小，为了与注意力头数量匹配
# BATCH_SIZE=128     # 批处理大小
# MAX_TOKENS=4096     # 模型生成的最大token数

# # ===================== 路径配置 ============================
# BASE_PATH="$(pwd)"  # 使用当前目录作为基础路径，如：/mnt/usercache/mengjinxiang/Project/SFT_and_Evaluation
# INFER_SCRIPT="${BASE_PATH}/tests/${TASK_NAME}.py"
# EVAL_SCRIPT="${BASE_PATH}/tests/eval/${TASK_NAME}_eval.py"

# # 自动生成输出文件路径
# PRED_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}.json"
# INFER_LOG="${BASE_PATH}/results/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_infer.log"
# EVAL_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_eval_results.json"

# # 确保输出目录存在
# mkdir -p "$(dirname "$PRED_OUTPUT")"
# mkdir -p "$(dirname "$INFER_LOG")"

# # ===================== 运行推理 ============================
# echo "Running inference with VLLM..."
# python "$INFER_SCRIPT" \
#     --model_path "$MODEL_PATH" \
#     --output_file "$PRED_OUTPUT" \
#     --log_file "$INFER_LOG" \
#     --base_path "$BASE_PATH" \
#     --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
#     --batch_size $BATCH_SIZE \
#     --max_tokens $MAX_TOKENS \
#     --temperature 0.0

# if [ $? -ne 0 ]; then
#   echo "Inference failed"
#   exit 1
# fi

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


# ===================== 参数读取 ============================
MODEL_PATH="/mnt/usercache/huggingface/Qwen2.5-3B-Instruct"   # 推理模型路径
EVAL_MODEL_PATH="/mnt/usercache/huggingface/Qwen2.5-7B-Instruct"  # 评估模型路径（用于LLM评估）
TASK_NAME="wikitq"  # 任务名，比如 tatqa, wikitq
TRAIN_TYPE="base"  # 训练方式，比如 grpo、ppo、sft
MODEL_SIZE="3b"    # 模型大小，比如 3b、7b
TENSOR_PARALLEL_SIZE=2  # 张量并行大小，为了与注意力头数量匹配
BATCH_SIZE=256     # 批处理大小
MAX_TOKENS=4096     # 模型生成的最大token数
USE_LLM_EVAL=true  # 是否使用LLM评估 (true/false)
LLM_EVAL_BATCH_SIZE=128  # LLM评估批大小

# ===================== 路径配置 ============================
BASE_PATH="$(pwd)"  # 使用当前目录作为基础路径
INFER_SCRIPT="${BASE_PATH}/tests/${TASK_NAME}.py"

# 根据是否使用LLM评估选择不同的评估脚本路径
if [ "$USE_LLM_EVAL" = true ]; then
  EVAL_SCRIPT="${BASE_PATH}/tests/llm_eval/${TASK_NAME}_eval.py"
else
  EVAL_SCRIPT="${BASE_PATH}/tests/eval/${TASK_NAME}_eval.py"
fi

# 自动生成输出文件路径
PRED_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}.json"
INFER_LOG="${BASE_PATH}/results/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_infer.log"

# 评估输出路径根据评估类型区分
if [ "$USE_LLM_EVAL" = true ]; then
  EVAL_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_llm_eval_results.json"
  EVAL_LOG="${BASE_PATH}/results/${TASK_NAME}/logs/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_llm_eval.log"
else
  EVAL_OUTPUT="${BASE_PATH}/results/${TASK_NAME}/${TASK_NAME}_${MODEL_SIZE}_${TRAIN_TYPE}_eval_results.json"
fi

# 确保输出目录存在
mkdir -p "$(dirname "$PRED_OUTPUT")"
mkdir -p "$(dirname "$INFER_LOG")"
mkdir -p "$(dirname "$EVAL_OUTPUT")"

# ===================== 运行推理 ============================
echo "Running inference with VLLM..."
python "$INFER_SCRIPT" \
    --model_path "$MODEL_PATH" \
    --output_file "$PRED_OUTPUT" \
    --log_file "$INFER_LOG" \
    --base_path "$BASE_PATH" \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --batch_size $BATCH_SIZE \
    --max_tokens $MAX_TOKENS \
    --temperature 0.0

if [ $? -ne 0 ]; then
  echo "Inference failed"
  exit 1
fi

# ===================== 运行评估 ============================
echo "Running evaluation..."

if [ "$USE_LLM_EVAL" = true ]; then
  # 使用LLM评估
  echo "Using LLM-based evaluation with model: $EVAL_MODEL_PATH"
  python "$EVAL_SCRIPT" \
      --results_file "$PRED_OUTPUT" \
      --output_file "$EVAL_OUTPUT" \
      --model_path "$EVAL_MODEL_PATH" \
      --log_file "$EVAL_LOG" \
      --base_path "$BASE_PATH" \
      --batch_size $LLM_EVAL_BATCH_SIZE \
      --tensor_parallel_size $TENSOR_PARALLEL_SIZE
else
  # 使用常规评估
  echo "Using standard evaluation"
  python "$EVAL_SCRIPT" \
      --results_file "$PRED_OUTPUT" \
      --output_file "$EVAL_OUTPUT" \
      --base_path "$BASE_PATH" 
fi

if [ $? -ne 0 ]; then
  echo "Evaluation failed"
  exit 1
fi

echo "Testing and evaluation completed successfully!"