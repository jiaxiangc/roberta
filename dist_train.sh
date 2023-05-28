
IP_PORT=10.132.57.20:13580
export OMP_NUM_THREADS=1

torchrun \
    --nnodes=1 \
    --nproc_per_node=2 \
    --rdzv_id=110 \
    --rdzv_backend=c10d \
    --rdzv_endpoint="127.0.0.1:13580" \
    train.py