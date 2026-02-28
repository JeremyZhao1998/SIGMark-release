import os
import random
import argparse
import numpy as np
from typing import List
from PIL import Image
from tqdm import tqdm

from diffusers.utils import export_to_video, load_video
from prompt_set import VBench2PromptSet


def parse_args():    
    parser = argparse.ArgumentParser()
    # Prompt set configuration
    parser.add_argument("--prompt_set", type=str, default="VBench2_aug", choices=["VBench2", "VBench2_aug", "VBench2_ch", "wanx_aug"])
    parser.add_argument("--image_prompt_dir", type=str, default="./prompt_set/VBench2_aug_img_prompt")
    # parser.add_argument("--image_prompt_dir", type=str, default=None)
    parser.add_argument("--num_prompts_per_dimension", type=int, default=5)
    parser.add_argument("--num_videos_per_prompt", type=int, default=4)
    parser.add_argument("--num_prompts_diversity", type=int, default=3)
    parser.add_argument("--num_videos_per_prompt_diversity", type=int, default=20)
    # Video generation/output configuration
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--num_frames", type=int, default=65)
    parser.add_argument("--fps", type=int, default=8)
    # Files output configuration
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--video_path", type=str, default="./outputs-1.0/HunyuanI2V-512x16/videoshield")
    parser.add_argument("--disturbances", type=list, nargs='+', default=["clip", "drop", "insert"])
    parsed_args = parser.parse_args()
    return parsed_args


def drop_frames(video, group_frame_num=6, drop_groups=5):
    """
    Randomly drop frames in groups:
    - Each group consists of consecutive frames of length group_frame_num
    - A total of drop_groups groups are dropped
    - Frame with index 0 is always kept
    - Returns: the video after dropping frames, and a list of kept frame indices
      referring to the original video
    """
    n = len(video)
    # Maximum number of non-overlapping groups that can fit after excluding frame 0
    max_groups = (n - 1) // group_frame_num
    if drop_groups > max_groups:
        raise ValueError(
            f"Cannot drop {drop_groups} groups: "
            f"at most {max_groups} non-overlapping groups are possible "
            f"for video length {n} and group_frame_num {group_frame_num}."
        )
    # Candidate group starting positions: from 1 to n - group_frame_num (inclusive)
    # Starting at 0 would include frame 0, so we start from 1
    candidate_starts = list(range(1, n - group_frame_num + 1))
    chosen_starts: List[int] = []
    while len(chosen_starts) < drop_groups and candidate_starts:
        start = random.choice(candidate_starts)
        chosen_starts.append(start)
        # Remove other candidate starts whose groups overlap with the chosen group
        new_candidates = []
        cur_start, cur_end = start, start + group_frame_num - 1
        for s in candidate_starts:
            s_start, s_end = s, s + group_frame_num - 1
            # Keep only intervals [s_start, s_end] that do not intersect [cur_start, cur_end]
            if s_end < cur_start or s_start > cur_end:
                new_candidates.append(s)
        candidate_starts = new_candidates
    if len(chosen_starts) < drop_groups:
        raise ValueError("Not enough non-overlapping groups could be selected.")
    # Collect all frame indices to drop
    dropped_indices = set()
    for s in chosen_starts:
        for i in range(s, s + group_frame_num):
            if 0 <= i < n:
                dropped_indices.add(i)
    # Build the kept indices list (in original order)
    kept_indices = [i for i in range(n) if i not in dropped_indices]
    # Frame 0 must be in kept_indices
    # assert 0 in kept_indices
    kept_video = [video[i] for i in kept_indices]
    return kept_video, kept_indices


# Helper to create a random noise frame with the same size and mode
# as the reference frame.
def _random_noise_frame_like(ref_frame):
    w, h = ref_frame.size
    mode = ref_frame.mode
    # Build a random array depending on image mode
    if mode == "L":
        # Grayscale
        arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    elif mode == "RGB":
        # 3-channel color
        arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    elif mode == "RGBA":
        # 4-channel color with alpha
        arr = np.random.randint(0, 256, (h, w, 4), dtype=np.uint8)
    else:
        # Generic fallback using number of bands
        bands = len(ref_frame.getbands())
        if bands == 1:
            arr = np.random.randint(0, 256, (h, w), dtype=np.uint8)
        else:
            arr = np.random.randint(0, 256, (h, w, bands), dtype=np.uint8)
    return Image.fromarray(arr, mode=mode)


def insert_frames(video, group_frame_num=6, insert_groups=5):
    """
    Randomly insert groups of noise frames into a video.

    - Each group consists of consecutive frames of length group_frame_num.
    - A total of insert_groups groups are inserted.
    - The position of the original frame at index 0 must remain unchanged
      (it stays at position 0 in the resulting video).
    - Inserted noise frames are marked with index -1 in the index list.
    - Returns:
        new_video: List of frames after insertion.
        index_map: For each frame in new_video, the index of the
                   corresponding original frame, or -1 for inserted noise.
    """
    n = len(video)
    # Gaps: there are (n + 1) possible insertion gaps:
    # gap 0: before frame 0
    # gap 1: between frame 0 and frame 1
    # ...
    # gap n: after frame n-1
    #
    # To keep frame 0 at position 0, we are NOT allowed to use gap 0.
    # So the allowed gaps are 1..n (inclusive).
    allowed_gaps = list(range(1, n + 1))
    if insert_groups > len(allowed_gaps):
        raise ValueError(
            f"Cannot insert {insert_groups} groups: "
            f"at most {len(allowed_gaps)} distinct insertion positions are possible "
            f"for video length {n}."
        )
    # Randomly choose distinct gaps where groups will be inserted
    chosen_gaps = sorted(random.sample(allowed_gaps, insert_groups))
    gap_set = set(chosen_gaps)
    new_video: List[Image.Image] = []
    index_map: List[int] = []
    ref_frame = video[0]
    # Iterate over original frames and insert noise groups at selected gaps.
    #
    # gap i: before frame i (for i in 1..n-1)
    # gap n: after the last frame
    # gap 0 is never used, so frame 0 stays at position 0.
    for i in range(n):
        gap_before_i = i  # gap index before frame i
        if gap_before_i in gap_set and gap_before_i != 0:
            # Insert a group of noise frames before frame i
            noise_frame = _random_noise_frame_like(ref_frame)
            for _ in range(group_frame_num):
                new_video.append(noise_frame)
                index_map.append(-1)
        # Append the original frame i
        new_video.append(video[i])
        index_map.append(i)
    # Handle gap n (after the last frame)
    if n in gap_set:
        for _ in range(group_frame_num):
            noise_frame = _random_noise_frame_like(ref_frame)
            new_video.append(noise_frame)
            index_map.append(-1)
    # Sanity check: original frame 0 is still at position 0
    # assert index_map[0] == 0
    return new_video, index_map


def clip_video(video, deleted_frame_num=30):
    """
    Randomly clip a segment from the video while keeping frame 0 fixed.

    Returns:
        clipped_video: List of frames after clipping.
        index_map: For each frame in clipped_video, its original index.
    """
    n = len(video)
    # We must keep at least frame 0, so we cannot delete n or more frames.
    if deleted_frame_num > n - 1:
        raise ValueError(
            f"Cannot delete {deleted_frame_num} frames from a video of length {n} "
            "while keeping frame 0."
        )
    # Length of the kept consecutive block (excluding frame 0)
    L = n - deleted_frame_num - 1
    # If L == 0, only keep frame 0
    if L == 0:
        clipped_video = [video[0]]
        index_map = [0]
        return clipped_video, index_map
    # L > 0: choose a random start index in [1, n - L]
    start = random.randint(1, n - L)
    end = start + L - 1
    # Build the result: frame 0 + frames [start..end]
    clipped_video = [video[0]] + video[start:end + 1]
    index_map = [0] + list(range(start, end + 1))
    # Sanity checks:
    # - index 0 stays at position 0
    # - all other kept indices form a consecutive range
    # assert index_map[0] == 0
    # assert index_map[1:] == list(range(start, end + 1))
    return clipped_video, index_map


def apply_disturbance(prompt_set, distortion_type):
    print(f"Applying disturbance: {distortion_type} for videos in {args.video_path}")
    output_path = args.video_path + f"_{distortion_type}"
    os.makedirs(output_path, exist_ok=True)
    os.system(f"cp {args.video_path}/*gt_watermark_messages.npz {output_path}/.")
    os.system(f"cp {args.video_path}/*maintained_info.pkl {output_path}/.")
    disturbance_info = {}
    for idx, (prompt, dimension, sample_name) in tqdm(enumerate(prompt_set), total=len(prompt_set)):
        video_path = os.path.join(args.video_path, dimension, sample_name + ".mp4")
        video = load_video(video_path)
        if distortion_type == "drop":
            disturbed_video, index_map = drop_frames(video, group_frame_num=6, drop_groups=5)
        elif distortion_type == "insert":
            disturbed_video, index_map = insert_frames(video, group_frame_num=6, insert_groups=5)
        elif distortion_type == "clip":
            disturbed_video, index_map = clip_video(video, deleted_frame_num=30)
        else:
            raise ValueError(f"Unknown disturbance type: {distortion_type}")
        disturbed_video_path = os.path.join(output_path, dimension)
        os.makedirs(disturbed_video_path, exist_ok=True)
        disturbed_video_path = os.path.join(disturbed_video_path, sample_name + ".mp4")
        export_to_video(disturbed_video, disturbed_video_path, fps=args.fps, quality=10)
        disturbance_info[f"{dimension}/{sample_name}.mp4"] = index_map
    # Save disturbance info
    np.savez_compressed(
        os.path.join(output_path, "disturbance_info.npz"),
        **disturbance_info
    )


def main():
    prompt_set = VBench2PromptSet(
        prompt_set_name=args.prompt_set,
        num_prompts_diversity=args.num_prompts_diversity, 
        num_videos_per_prompt=args.num_videos_per_prompt,
        num_prompts_per_dimension=args.num_prompts_per_dimension,
        num_videos_per_prompt_diversity=args.num_videos_per_prompt_diversity
    )
    for distortion in args.disturbances:
        apply_disturbance(prompt_set, distortion)


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    main()
