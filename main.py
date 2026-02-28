import os
import random
import argparse
import numpy as np
import logging
import time

import torch
from torch.utils.data import DataLoader, DistributedSampler
from diffusers import HunyuanVideoPipeline, HunyuanVideoImageToVideoPipeline
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers.utils import export_to_video, load_video, load_image

from watermarks import VideoGenWatermarkBase, VideoShieldWatermark, VideoMarkWatermark, SIGMarkWatermark
from models.schedulers import FlowMatchEulerDiscreteInverseScheduler
from prompt_set import VBench2PromptSet
from utils.video_utils import align_frames
from utils.distributed_utils import init_distributed_mode, get_rank, reduce_dict, all_gather


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
    parser.add_argument("--small_scale_test", type=int, default=-1)
    # Video generation/output configuration
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--num_steps", type=int, default=50)
    # Video watermark configuration
    parser.add_argument("--watermark_method", type=str, default="sigmark", choices=["none", "videoshield", "videomark", "sigmark"])
    parser.add_argument("--ch_factor", type=int, default=2)
    parser.add_argument("--hw_factor", type=int, default=8)
    parser.add_argument("--fr_factor", type=int, default=1)
    parser.add_argument("--sgo", type=int, default=0)
    parser.add_argument("--of_seg", type=int, default=1)
    parser.add_argument("--sw_det", type=int, default=1)
    # Files output configuration
    parser.add_argument("--output_path", type=str, default="./outputs-1.0/HunyuanI2V-512x16/sigmark")
    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _check_settings(args, pipeline=None, loader=None):
    if loader is not None:
        total_batch_size = loader.batch_size
        if args.distributed:
            total_batch_size *= args.world_size
        assert args.num_videos_per_prompt % total_batch_size == 0, \
            f"num_videos_per_prompt={args.num_videos_per_prompt} should be " \
            f"a multiple of total batch_size={total_batch_size}"
        assert args.num_videos_per_prompt_diversity % total_batch_size == 0, \
            f"num_videos_per_prompt_diversity={args.num_videos_per_prompt_diversity} "\
            f"should be a multiple of total batch_size={total_batch_size}"
    if pipeline is not None:
        assert args.num_frames % pipeline.vae_scale_factor_temporal == 1, \
            f"Number of frames must be vae_scale_factor_temporal * n + 1, " \
            f"got {args.num_frames} frames and vae_scale_factor_temporal=" \
            f"{pipeline.vae_scale_factor_temporal}"
        assert args.width % pipeline.vae_scale_factor_spatial == 0 and \
            args.height % pipeline.vae_scale_factor_spatial == 0, \
            f"Width and height must be a multiple of vae_scale_factor_spatial, "\
            f"got {args.width}x{args.height} and " \
            f"vae_scale_factor_spatial={pipeline.vae_scale_factor_spatial}"


def build_dataloader(args):
    prompt_set = VBench2PromptSet(
        prompt_set_name=args.prompt_set,
        num_prompts_diversity=args.num_prompts_diversity, 
        num_videos_per_prompt=args.num_videos_per_prompt,
        num_prompts_per_dimension=args.num_prompts_per_dimension,
        num_videos_per_prompt_diversity=args.num_videos_per_prompt_diversity
    )
    if args.distributed:
        sampler = DistributedSampler(prompt_set, shuffle=False, drop_last=False)
    else:
        sampler = None
    prompt_loader = DataLoader(
        dataset=prompt_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        drop_last=False,
        pin_memory=True,
        sampler=sampler
    )
    _check_settings(args, loader=prompt_loader)
    return prompt_loader


@torch.no_grad()
def build_pipeline(args, model_id, inverse=False):
    start = time.time()
    if "Hunyuan" in model_id:
        pipeline_quant_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_8bit",
            quant_kwargs={"load_in_8bit": True},
            components_to_quantize=["text_encoder", "text_encoder_2"]
        ) if args.quant_text_encoder else None
        if "I2V" in model_id:
            pipeline = HunyuanVideoImageToVideoPipeline.from_pretrained(
                model_id, quantization_config=pipeline_quant_config, torch_dtype=args.dtype, device_map="cuda")
        else:
            pipeline = HunyuanVideoPipeline.from_pretrained(
                model_id, quantization_config=pipeline_quant_config, torch_dtype=args.dtype, device_map="cuda")
        if args.distributed and args.rank != 0:
            pipeline.set_progress_bar_config(disable=True)
        if inverse:
            inverse_scheduler = FlowMatchEulerDiscreteInverseScheduler.from_pretrained(
                model_id,
                subfolder='scheduler',
            )
            pipeline.scheduler = inverse_scheduler
    else:
        raise NotImplementedError(f"model {model_id} is not supported yet")
    # pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_tiling()
    _check_settings(args, pipeline)
    print(f"Load pipeline finished, time cost: {time.time() - start}s")
    return pipeline


def build_watermark(args, pipeline=None, maintained_info_path=None):    
    kwargs = {
        "video_h": args.height,
        "video_w": args.width,
        "video_f": args.num_frames,
        "latent_c": pipeline.transformer.config.in_channels if pipeline is not None else 16,
        "vae_scale_factor_spatial": pipeline.vae_scale_factor_spatial if pipeline is not None else 8,
        "vae_scale_factor_temporal": pipeline.vae_scale_factor_temporal if pipeline is not None else 4,
        "ch_factor": args.ch_factor,
        "hw_factor": args.hw_factor,
        "fr_factor": args.fr_factor,
        "maintained_info_path": maintained_info_path,
        "batch_size": args.batch_size,
        "seed": args.seed if args.debug else None,
        "dtype": args.dtype,
        "device": args.device
    }
    if args.watermark_method == "none":
        return VideoGenWatermarkBase(**kwargs)
    if args.watermark_method == "videoshield":
        return VideoShieldWatermark(**kwargs)
    if args.watermark_method == "videomark":
        return VideoMarkWatermark(**kwargs)
    if args.watermark_method == "sigmark":
        return SIGMarkWatermark(**kwargs)
    else:
        raise NotImplementedError(f"watermark method {args.watermark_method} is not supported yet")


@torch.no_grad()
def encode_videos(args, pipeline, videos):
    frame_tensors = pipeline.video_processor.preprocess_video(videos)
    latents = pipeline.vae.encode(frame_tensors.to(pipeline.vae.dtype).to(args.device)).latent_dist.mode()
    latents *= pipeline.vae.config.scaling_factor
    return latents


@torch.no_grad()
def decode_videos(pipeline, latents):
    latents = latents.to(pipeline.vae.dtype) / pipeline.vae.config.scaling_factor
    videos = pipeline.vae.decode(latents, return_dict=False)[0]
    video_frames = pipeline.video_processor.postprocess_video(videos, output_type="pil")
    return video_frames


def get_watermark_len(args, pipeline=None):
    video_h, video_w, video_f = args.height, args.width, args.num_frames
    latent_c = pipeline.transformer.config.in_channels if pipeline is not None else 16
    vae_scale_factor_spatial = pipeline.vae_scale_factor_spatial if pipeline is not None else 8
    vae_scale_factor_temporal = pipeline.vae_scale_factor_temporal if pipeline is not None else 4
    ch_factor, hw_factor, fr_factor = args.ch_factor, args.hw_factor, args.fr_factor
    latent_h, latent_w = video_h // vae_scale_factor_spatial, video_w // vae_scale_factor_spatial
    latent_f = (video_f - 1) // vae_scale_factor_temporal
    watermark_h, watermark_w = latent_h // hw_factor, latent_w // hw_factor
    watermark_f = latent_f // fr_factor
    watermark_c = latent_c // ch_factor
    watermark_len = watermark_h * watermark_w * watermark_f * watermark_c
    return watermark_len


def get_setting_brief_str(args, pipeline=None):
    setting_brief_str = f"{args.model_name}-{args.prompt_set}-{args.width}x{args.height}-{args.num_frames}frams"
    if args.watermark_method != "None":
        setting_brief_str += f"-{args.watermark_method}-{get_watermark_len(args, pipeline)}bits"
    return setting_brief_str


@torch.no_grad()
def generate_videos(args):
    prompt_loader = build_dataloader(args)
    pipeline = build_pipeline(args, os.path.join(args.model_base_path, args.model_name))
    setting_brief_str = get_setting_brief_str(args, pipeline)
    maintained_info_path = os.path.join(args.output_path, setting_brief_str + "-maintained_info.pkl")
    watermarking = build_watermark(
        args, pipeline,
        maintained_info_path=maintained_info_path if os.path.exists(maintained_info_path) else None
    )
    gt_messages_path = os.path.join(args.output_path, setting_brief_str + "-gt_watermark_messages.npz")
    gt_watermark_messages = {} if not os.path.exists(gt_messages_path) else dict(np.load(gt_messages_path))
    for batch_idx, (prompts, dimensions, sample_names) in enumerate(prompt_loader):
        if args.small_scale_test > 0 and batch_idx > args.small_scale_test:
            break
        prompt, dimension = prompts[0], dimensions[0]
        if not args.distributed or args.rank == 0:
            print(f"[{batch_idx + 1}/{len(prompt_loader)}]: Generating videos for {prompt[:180]}...")
        video_dir = os.path.join(args.output_path, dimension)
        video_exist_check = [os.path.exists(os.path.join(video_dir, sample_name + ".mp4")) for sample_name in sample_names]
        video_exist_check = [v for sub in all_gather(video_exist_check) for v in sub] if args.distributed else video_exist_check
        if all(video_exist_check):
            if not args.distributed or args.rank == 0:
                print(f"[{batch_idx + 1}/{len(prompt_loader)}]: " 
                    f"Videos already exist for {setting_brief_str}, {dimension}/{prompt[:180]} . ")
            continue
        os.makedirs(video_dir, exist_ok=True)
        image_prompt = load_image(os.path.join(args.image_prompt_dir, dimension, prompt[:180] + "-0.png")) \
            if args.image_prompt_dir is not None else None
        watermark_messages = watermarking.generate_random_watermark_message()
        gt_messages_batch = {f"{dimension}/{sample_names[i]}": watermark_messages[i] for i in range(args.batch_size)}
        if args.distributed:
            gt_messages_batch = reduce_dict(gt_messages_batch)
        gt_watermark_messages.update(gt_messages_batch)
        init_latents_watermarked = watermarking.create_watermarked_latents(watermark_messages)
        with torch.inference_mode(), torch.autocast("cuda", dtype=args.dtype):
            kwargs = {
                "prompt": [prompt] * args.batch_size,
                "height": args.height,
                "width": args.width,
                "num_frames": args.num_frames,
                "latents": init_latents_watermarked,
                "guidance_scale": args.guidance_scale,
                "num_inference_steps": args.num_steps,
                "output_type": "latent",
                "return_dict": False
            }
            if image_prompt is not None:
                kwargs["image"] = [image_prompt] * args.batch_size
            video_latents = pipeline(**kwargs)[0]
            videos = decode_videos(pipeline, video_latents)
        for i, video in enumerate(videos):
            export_to_video(video, os.path.join(video_dir, sample_names[i] + ".mp4"), fps=args.fps, quality=10)
        if not args.distributed or args.rank == 0:
            watermarking.save_maintained_info(maintained_info_path)
            if args.watermark_method != "None":
                np.savez(gt_messages_path, **gt_watermark_messages)
            print(f"[{batch_idx + 1}/{len(prompt_loader)}]: " 
                  f"Video generation for {setting_brief_str}, {dimension}/{prompt[:180]} finished. ")
        if args.distributed:
            torch.distributed.barrier()


def frame_watermark_idx_map(frame_idx_list, vae_scale_factor_temporal=4):
    watermark_idx_list = []
    for idx in frame_idx_list:
        if idx > 0:
            watermark_idx_list.append((idx - 1) // vae_scale_factor_temporal)
    return list(set(sorted(watermark_idx_list)))


def calculate_bit_acc(gt_message, extracted_message, valid_idx=None):
    # valid_idx controls the frames used for bit accuracy calculation
    if valid_idx is not None:
        # Only keep the watermark of the valid frames
        gt_message = gt_message[:, valid_idx, :, :]
        extracted_message = extracted_message[:, valid_idx, :, :]
    bit_acc = (gt_message == extracted_message).astype(np.float32).mean()
    return bit_acc


def extract(args):
    prompt_loader = build_dataloader(args)
    pipeline = build_pipeline(args, os.path.join(args.model_base_path, args.model_name), inverse=True)
    setting_brief_str = get_setting_brief_str(args, pipeline)
    maintained_info_path = os.path.join(args.output_path, setting_brief_str + "-maintained_info.pkl")
    watermarking = build_watermark(args, pipeline, maintained_info_path)
    gt_messages_path = os.path.join(args.output_path, setting_brief_str + "-gt_watermark_messages.npz")
    gt_watermark_messages = dict(np.load(gt_messages_path))
    extracted_messages_path = os.path.join(args.output_path, setting_brief_str + "-extracted_messages.npz")
    extracted_messages_all = {} if not os.path.exists(extracted_messages_path) else dict(np.load(extracted_messages_path))
    bit_accuracy_path = os.path.join(args.output_path, setting_brief_str + "-bit_accuracy.npz")
    bit_accuracy_all = {} if not os.path.exists(bit_accuracy_path) else dict(np.load(bit_accuracy_path))
    disturbance_info_path = os.path.join(args.output_path, "disturbance_info.npz")
    disturbance_info = None if not os.path.exists(disturbance_info_path) else dict(np.load(disturbance_info_path))
    for batch_idx, (prompts, dimensions, sample_names) in enumerate(prompt_loader):
        if args.small_scale_test > 0 and batch_idx > args.small_scale_test:
            break
        prompt, dimension = prompts[0], dimensions[0]
        video_check = [f"{dimension}/{sample_name}" in gt_watermark_messages for sample_name in sample_names]
        video_check = [v for sub in all_gather(video_check) for v in sub] if args.distributed else video_check
        if not all(video_check):
            if not args.distributed or args.rank == 0:
                print(f"[{batch_idx + 1}/{len(prompt_loader)}]: " 
                    f"Videos do not exist for {dimension}/{prompt[:180]} , skip.")
            continue
        if not args.distributed or args.rank == 0:
            print(f"[{batch_idx + 1}/{len(prompt_loader)}]: "
                  f"Extracting watermark messages for {prompt[:180]}...")
        sample_names_all = [v for sub in all_gather(sample_names) for v in sub] if args.distributed else sample_names
        bit_acc_exist_check = [f"{dimension}/{sample_name}" in bit_accuracy_all for sample_name in sample_names_all]
        if all(bit_acc_exist_check):
            if not args.distributed or args.rank == 0:
                for sample_name in sample_names_all:
                    k, v = f"{dimension}/{sample_name}", bit_accuracy_all[f"{dimension}/{sample_name}"]
                    print(f"[{batch_idx + 1}/{len(prompt_loader)}]: {k}.mp4, bit acc: {v}.")
            continue
        video_dir = os.path.join(args.output_path, dimension)
        videos = [load_video(os.path.join(video_dir, sample_name + ".mp4")) for sample_name in sample_names]
        if disturbance_info is not None:
            # Apply SGO module or other frame recovery operations before extraction
            index_maps = [disturbance_info[f"{dimension}/{sample_name}.mp4"].tolist() for sample_name in sample_names]
            valid_index = [frame_watermark_idx_map(index_map) for index_map in index_maps]
            if not args.sgo or args.watermark_method != "sigmark":
                videos = [align_frames(video, args.num_frames) for video in videos]
            else:
                videos = [
                    watermarking.segment_group_ordering(
                        video, args.num_frames, args.of_seg, args.sw_det, pipeline)
                    for video in videos
                ]
        else:
            valid_index = None
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
        extracted_watermark_messages = watermarking.extract_watermark(inverted_init_latents)
        extracted_m_batch, bit_acc_batch = {}, {}
        for extracted_m, sample_name, valid_idx in zip(extracted_watermark_messages, sample_names, valid_index):
            extracted_m_batch[f"{dimension}/{sample_name}"] = extracted_m
            gt_m = gt_watermark_messages[f"{dimension}/{sample_name}"]
            bit_acc = calculate_bit_acc(gt_m, extracted_m)
            bit_acc_batch[f"{dimension}/{sample_name}"] = bit_acc
        if args.distributed:
            extracted_m_batch = reduce_dict(extracted_m_batch)
            bit_acc_batch = reduce_dict(bit_acc_batch)
        if not args.distributed or args.rank == 0:
            for k, v in bit_acc_batch.items():
                print(f"[{batch_idx + 1}/{len(prompt_loader)}]: {k}.mp4, bit acc: {v}.")
        extracted_messages_all.update(extracted_m_batch)
        bit_accuracy_all.update(bit_acc_batch)
        if not args.distributed or args.rank == 0:
            np.savez(extracted_messages_path, **extracted_messages_all)
            np.savez(bit_accuracy_path, **bit_accuracy_all)
            bit_acc_value_tmp = [v for _, v in bit_accuracy_all.items()]
            print(f"[{batch_idx + 1}/{len(prompt_loader)}] finished, "
                  f"avg bit acc so far: {sum(bit_acc_value_tmp) / len(bit_acc_value_tmp)}.")
        if args.distributed:
            torch.distributed.barrier()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("diffusers").setLevel(logging.ERROR)
    parsed_args = parse_args()
    parsed_args.device = torch.device('cuda')
    torch.cuda.set_device(get_rank())
    set_random_seed(parsed_args.seed + get_rank())
    torch.backends.cudnn.benchmark = False
    torch.set_grad_enabled(False)
    init_distributed_mode(parsed_args)
    if not parsed_args.distributed or parsed_args.rank == 0:
        for k, v in parsed_args.__dict__.items():
            print(f"{k}: {v}")
        print("------------------------------------")
    if parsed_args.precision == "fp16":
        parsed_args.dtype = torch.float16
    elif parsed_args.precision == "bf16":
        parsed_args.dtype = torch.bfloat16
    elif parsed_args.precision == "fp32":
        parsed_args.dtype = torch.float32
    else:
        raise NotImplementedError(f"precision: {parsed_args.precision} not implemented")
    if parsed_args.mode == "gen":
        generate_videos(parsed_args)
    elif parsed_args.mode == "extract":
        extract(parsed_args)
    if parsed_args.distributed:
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
