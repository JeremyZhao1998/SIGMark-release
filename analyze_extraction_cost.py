import os
import sys
import argparse
import numpy as np
import logging
import random
import time

import torch
from diffusers.utils import load_video

from main import (
    set_random_seed, 
    init_distributed_mode,
    build_pipeline, 
    build_watermark, 
    get_setting_brief_str, 
    encode_videos
)
from watermarks import VideoShieldWatermark, VideoMarkWatermark, SIGMarkWatermark
from prompt_set import VBench2PromptSet


def parse_args():    
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="extract", choices=["gen", "extract"])
    # Model configuration
    parser.add_argument("--model_base_path", type=str, default="/dfs/data/pretrained_models")
    parser.add_argument("--model_name", type=str, default="HunyuanVideo-I2V-community")
    parser.add_argument("--precision", type=str, default="bf16", choices=["fp16", "bf16", "fp32"])
    parser.add_argument("--quant_text_encoder", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--debug", type=int, default=0, 
                        help="Debug mode, 1 for fix random seed in encryption for reproducibility.")
    # Prompt set configuration
    parser.add_argument("--prompt_set", type=str, default="VBench2_aug", choices=["VBench2", "VBench2_aug", "VBench2_ch", "wanx_aug"])
    parser.add_argument("--image_prompt_dir", type=str, default="./prompt_set/VBench2_aug_img_prompt")
    # parser.add_argument("--image_prompt_dir", type=str, default=None)
    parser.add_argument("--num_prompts_per_dimension", type=int, default=5)
    parser.add_argument("--num_videos_per_prompt", type=int, default=4)
    parser.add_argument("--num_prompts_diversity", type=int, default=3)
    parser.add_argument("--num_videos_per_prompt_diversity", type=int, default=20)
    # Video generation/output configuration
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_steps", type=int, default=50)
    # Video watermark configuration
    parser.add_argument("--watermark_methods", type=list, default=["videoshield", "sigmark"])
    parser.add_argument("--ch_factor", type=int, default=2)
    parser.add_argument("--hw_factor", type=int, default=16)
    parser.add_argument("--fr_factor", type=int, default=4)
    parser.add_argument("--sgo", type=int, default=0)
    parser.add_argument("--of_seg", type=int, default=0)
    parser.add_argument("--sw_det", type=int, default=0)
    # Files output configuration
    parser.add_argument("--output_path", type=str, default="./outputs/HunyuanI2V-128x4")
    parsed_args = parser.parse_args()
    return parsed_args


def get_size(element_list):
    size = 0
    for element in element_list:
        if isinstance(element, np.ndarray):
            size += element.nbytes + sys.getsizeof(element)
        elif isinstance(element, torch.Tensor):
            size += element.element_size() * element.nelement() + sys.getsizeof(element)
        else:
            size += sys.getsizeof(element)
    return size


@torch.no_grad()
def main(args, watermark_method="sigmark"):
    args.watermark_method = watermark_method
    setting_brief_str = get_setting_brief_str(args, pipeline)
    maintained_info_path = os.path.join(args.output_path, watermark_method, setting_brief_str + "-maintained_info.pkl")
    watermarking = build_watermark(args, pipeline, maintained_info_path)
    gt_messages_path = os.path.join(args.output_path, watermark_method, setting_brief_str + "-gt_watermark_messages.npz")
    gt_watermark_messages = dict(np.load(gt_messages_path))
    watermarking.device = "cpu"  # Performing extraction operations all on CPU to ensure fair comparison
    total_video_nums = [100, 200, 400, 800, 1600, 3200]
    # Prepare the simulated total number of videos
    max_num_videos = max(total_video_nums)
    if isinstance(watermarking, SIGMarkWatermark):
        pass # SIGMarkWatermark is a blind watermark which does not need to maintain per-video info
    elif isinstance(watermarking, VideoShieldWatermark):
        watermarking.encrypted_messages = watermarking.encrypted_messages.cpu()
        added_num = max_num_videos - len(watermarking.encrypted_messages)
        rand_template = torch.randint_like(watermarking.encrypted_messages[0], low=0, high=2)
        encrypted_messages = torch.cat([
            watermarking.encrypted_messages, 
            rand_template.unsqueeze(0).repeat(added_num, 1, 1, 1, 1)
        ], dim=0)
        keys = watermarking.keys + [np.random.randint(0, 256, 32, dtype=np.uint8).tobytes() for _ in range(added_num)]
        nonces = watermarking.nonces + [np.random.randint(0, 256, 12, dtype=np.uint8).tobytes() for _ in range(added_num)]
    elif isinstance(watermarking, VideoMarkWatermark):
        added_num = max_num_videos - len(watermarking.maintained_info["watermark_messages"])
        watermark_messages = watermarking.maintained_info["watermark_messages"] + [
            np.random.rand(*watermarking.maintained_info["watermark_messages"][0].shape) 
            for _ in range(added_num)
        ]
    print()
    print("=======================================================================================")
    print("This script tests the time cost of watermark extraction of: ")
    print(f"Watermark method: {args.watermark_method.upper()} on prompt set: {args.prompt_set}")
    print(f"We simulate the siduation that under a total number of {total_video_nums} videos, ")
    print("the watermark from a SINGLE video is to be extracted, and report the space and time cost.")
    print("With the growth of the total number of videos, the time cost of extracting a single video")
    print("should remain unchanged if the watermarking method is scalable.")
    print("We perform all the extraction operations across all watermarkings on CPU to ensure fair comparison.")
    print("=======================================================================================")
    for total_video_num in total_video_nums:
        sample_idx = random.randint(0, min(len(prompt_set), total_video_num) - 1)
        prompt, dimension, sample_name = prompt_set[sample_idx]
        video_dir = os.path.join(args.output_path, watermark_method, dimension)
        videos = [load_video(os.path.join(video_dir, sample_name + ".mp4"))]
        print(f"Tracing time cost and memory cost of {args.watermark_method} watermark extraction")
        print(f"under total video num: {total_video_num}.")
        # Adjust the total number of videos in the watermarking module
        if isinstance(watermarking, SIGMarkWatermark):
            maintained_info_size = 0
            for encoding_key in watermarking.maintained_info["encoding_keys"]:
                maintained_info_size += get_size(encoding_key)
            for decoding_key in watermarking.maintained_info["decoding_keys"]:
                maintained_info_size += get_size(decoding_key)
        elif isinstance(watermarking, VideoShieldWatermark):
            watermarking.encrypted_messages = encrypted_messages[:total_video_num]
            watermarking.keys = keys[:total_video_num]
            watermarking.nonces = nonces[:total_video_num]
            maintained_info_size = get_size([watermarking.encrypted_messages]) + \
                get_size(watermarking.keys) + get_size(watermarking.nonces)
        elif isinstance(watermarking, VideoMarkWatermark):
            watermarking.maintained_info["watermark_messages"] = watermark_messages[:total_video_num]
            maintained_info_size = get_size(watermarking.maintained_info["watermark_messages"]) + \
                get_size(watermarking.encoding_key) + get_size(watermarking.decoding_key)
        else:
            raise NotImplementedError(f"watermarking method {args.watermark_method} not implemented")
        print(f"Maintained info size: {maintained_info_size} bytes ({maintained_info_size / 1024 / 1024:.2f} MB).")
        start_time = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=args.dtype):
            inverted_video_latents = encode_videos(args, pipeline, videos)
            kwargs = {
                "prompt": [""] * args.batch_size,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_steps,
                "num_videos_per_prompt": args.batch_size,
                "num_frames": args.num_frames,
                "latents": inverted_video_latents,
                "output_type": "latent",
                "return_dict": False
            }
            if "I2V" in args.model_name:
                kwargs.update({
                    "image": [videos[0][0]] * args.batch_size,
                    "height": videos[0][0].height,
                    "width": videos[0][0].width,
                })
            inverted_init_latents = pipeline(**kwargs)[0]
        inversion_time = time.time()
        inversion_time_cost = inversion_time - start_time
        print(f"Inversion time cost: {inversion_time_cost:.3f} seconds.")
        extracted_watermark_messages = watermarking.extract_watermark(inverted_init_latents.cpu())
        extraction_time = time.time()
        extraction_time_cost = extraction_time - inversion_time
        gt_message = gt_watermark_messages[f"{dimension}/{sample_name}"]
        bit_acc = (extracted_watermark_messages[0] == gt_message).astype(np.float32).mean()
        print(f"Bit accuracy of extracted watermark message: {bit_acc * 100:.2f} %.")
        print(f"Extraction time cost: {extraction_time_cost:.3f} seconds.")
        total_time_cost = extraction_time - start_time
        print(f"Total time cost: {total_time_cost:.3f} seconds.")
        print("-----------------------------------------------------------------------------------")
    pass


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    parsed_args = parse_args()
    parsed_args.device = torch.device('cuda')
    set_random_seed(parsed_args.seed)
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    init_distributed_mode(parsed_args)
    for k, v in parsed_args.__dict__.items():
        print(f"{k}: {v}")
    print("-----------------------------------------------------------------------------------")
    if parsed_args.precision == "fp16":
        parsed_args.dtype = torch.float16
    elif parsed_args.precision == "bf16":
        parsed_args.dtype = torch.bfloat16
    elif parsed_args.precision == "fp32":
        parsed_args.dtype = torch.float32
    else:
        raise NotImplementedError(f"precision: {parsed_args.precision} not implemented")
    pipeline = build_pipeline(
        parsed_args, 
        os.path.join(parsed_args.model_base_path, parsed_args.model_name), inverse=True
    )
    prompt_set = VBench2PromptSet(
        prompt_set_name=parsed_args.prompt_set,
        num_prompts_diversity=parsed_args.num_prompts_diversity, 
        num_videos_per_prompt=parsed_args.num_videos_per_prompt,
        num_prompts_per_dimension=parsed_args.num_prompts_per_dimension,
        num_videos_per_prompt_diversity=parsed_args.num_videos_per_prompt_diversity
    )
    for watermark in parsed_args.watermark_methods:
        main(parsed_args, watermark)
