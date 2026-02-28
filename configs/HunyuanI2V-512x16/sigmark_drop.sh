source /dfs/data/miniconda3/bin/activate
cd /dfs/data/SIGMark
conda activate sigmark

N_GPUS=1
BATCH_SIZE=1
MODEL_PATH=/dfs/data/pretrained_models
OUTPUT_DIR=./outputs-1.0/HunyuanI2V-512x16/sigmark_drop

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun \
--nproc_per_node=${N_GPUS} \
main.py \
--mode=extract \
--model_base_path=${MODEL_PATH} \
--model_name=HunyuanVideo-I2V-community \
--prompt_set=VBench2_aug \
--watermark_method=sigmark \
--sgo=1 \
--ch_factor=2 \
--hw_factor=8 \
--fr_factor=1 \
--batch_size=${BATCH_SIZE} \
--output_path=${OUTPUT_DIR}
