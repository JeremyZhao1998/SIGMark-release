source /dfs/data/miniconda3/bin/activate
cd /dfs/data/SIGMark
conda activate sigmark

N_GPUS=4
BATCH_SIZE=4
MODEL_PATH=/root/private_data/model
OUTPUT_DIR=./outputs/HunyuanT2V-512/videomark

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
--nproc_per_node=${N_GPUS} \
main.py \
--mode=gen \
--model_base_path=${MODEL_PATH} \
--model_name=HunyuanVideo-community \
--prompt_set=VBench2_aug \
--watermark_method=videomark \
--ch_factor=2 \
--hw_factor=8 \
--fr_factor=16 \
--batch_size=${BATCH_SIZE} \
--output_path=${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
--nproc_per_node=${N_GPUS} \
main.py \
--mode=extract \
--model_base_path=${MODEL_PATH} \
--model_name=HunyuanVideo-community \
--prompt_set=VBench2_aug \
--watermark_method=videomark \
--ch_factor=2 \
--hw_factor=8 \
--fr_factor=16 \
--batch_size=${BATCH_SIZE} \
--output_path=${OUTPUT_DIR}
