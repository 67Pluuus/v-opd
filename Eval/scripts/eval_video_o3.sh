source ~/.bashrc

echo "Debug: Current working directory: $(pwd)"

echo "Debug: All received parameters:"
echo "  \$1 (DATA): '${1}'"
echo "  \$2 (MODEL): '${2}'"
echo "  \$3 (MODE): '${3}'"
echo "  \$4 (REUSE): '${4}'"
echo "  \$5 (NPROC_PER_NODE): '${5}'"
echo "  \$6 (WORK_DIR): '${6}'"
echo "  Total number of parameters: $#"

DATA=${1:-"Video-MME_2fps"}
MODEL=${2:-"Video-o3"}
MODE=${3:-"all"}
REUSE_FLAG=${4:-""}
NPROC_PER_NODE=${5:-"4"}
WORK_DIR=${6:-"./outputs"}

conda activate eval_video_o3

export OMP_NUM_THREADS=1
export DISABLE_ADDMM_CUDA_LT=1
export TORCH_CUDNN_USE_HEURISTIC_MODE_B=1

export OMP_NUM_THREADS=1 
export MKL_NUM_THREADS=1 
export OPENBLAS_NUM_THREADS=1 
export NUMEXPR_NUM_THREADS=1 
export TORCH_NUM_THREADS=1

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN

export NCCL_BLOCKING_WAIT=1

export NCCL_SOCKET_IFNAME=bond0

export NCCL_IB_HCA=mlx5_0

# P2P 通信优化
export NCCL_P2P_LEVEL=NVL

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True



CMD="torchrun --nproc-per-node=${NPROC_PER_NODE} run.py --data ${DATA} --model ${MODEL} --mode ${MODE} --work-dir ${WORK_DIR}"

if [ -n "${REUSE_FLAG}" ] && [ "${REUSE_FLAG}" = "--reuse" ]; then
    CMD="${CMD} --reuse"
fi

echo "Run command: ${CMD}"
${CMD}