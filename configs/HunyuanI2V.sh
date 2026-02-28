source /dfs/data/miniconda3/bin/activate
cd /dfs/data/SIGMark
conda activate sigmark

N_GPUS=4
BATCH_SIZE=4
MODEL_PATH=/dfs/data/pretrained_models
OUTPUT_DIR=./outputs/HunyuanI2V

CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=4 torchrun \
--nproc_per_node=${N_GPUS} \
main.py \
--mode=gen \
--model_base_path=${MODEL_PATH} \
--model_name=HunyuanVideo-I2V-community \
--prompt_set=VBench2_aug \
--image_prompt_dir=./prompt_set/VBench2_aug_img_prompt \
--watermark_method=none \
--batch_size=${BATCH_SIZE} \
--output_path=${OUTPUT_DIR}
