NUM_GPUS=8

DATA_LIST=(
    'Video-MME_2fps_limit_768'
    'MLVU_2fps_limit_768'
    'LVBench_2fps_limit_768'
    'LongVideoBench_2fps_limit_768'
    'VideoMMMU_2fps_limit_768'
    'MMVU_2fps_limit_768'
    'Video_Holmes_2fps_limit_768'
    'Charades_2fps_limit_768' # corresponds to video-o3-tg
) 

MODEL_LIST=(
    'video-o3'
    # 'video-o3-tg'
)

MODE='all'
REUSE='--reuse' 
NPROC_PER_NODE=${NUM_GPUS}

LOG_DIR="./runlogs"
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date +"%Y%m%d_%H")


for DATA in "${DATA_LIST[@]}"; do
    for MODEL in "${MODEL_LIST[@]}"; do

        if [[ "$DATA" == "Charades_2fps_limit_768" && "$MODEL" == "video-o3" ]]; then
            MODEL="video-o3-tg"
        fi

        LOG_SUB_DIR="${LOG_DIR}/${MODEL}"
        mkdir -p "$LOG_SUB_DIR"
        LOG_FILE="${LOG_SUB_DIR}/${DATA}_${NPROC_PER_NODE}GPUs_${TIMESTAMP}.txt"
        WORK_DIR="${LOG_SUB_DIR}/outputs"
        mkdir -p "$WORK_DIR"

        echo "=========================================="
        echo "Processing dataset: $DATA"
        echo "Using model: $MODEL"
        echo "Log file: $LOG_FILE"
        echo "Output will be shown in the terminal and saved to the log file"
        echo "=========================================="

        ARGS=("${DATA}" "${MODEL}" "${MODE}" "${REUSE}" "${NPROC_PER_NODE}" "${WORK_DIR}")
        bash scripts/eval_video_o3.sh "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"

        echo "Task for dataset $DATA with model $MODEL has been completed. Log saved to: $LOG_FILE"
        echo "Sleeping for 1.0 second"
        sleep 1.0
    done
done

echo "All tasks have been completed!"
echo "A total of $((${#DATA_LIST[@]} * ${#MODEL_LIST[@]})) tasks were executed"
