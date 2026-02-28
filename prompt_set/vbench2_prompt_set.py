import os
import json
from tqdm import tqdm

import torch


class VBench2PromptSet(torch.utils.data.Dataset):

    def __init__(self, 
                 prompt_set_name,
                 num_prompts_per_dimension=None,
                 num_videos_per_prompt=4,
                 num_prompts_diversity=None,
                 num_videos_per_prompt_diversity=20):
        self.prompt_set_name = prompt_set_name
        self.num_videos_per_prompt = num_videos_per_prompt
        self.num_videos_per_prompt_diversity = num_videos_per_prompt_diversity
        # Load prompts from the directory
        self.meta_info_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meta_info")
        assert prompt_set_name in ["VBench2", "VBench2_aug", "VBench2_ch", "wanx_aug"], \
            f"Unknown prompt set name: {prompt_set_name}, expected one of [VBench2, VBench2_aug, VBench2_ch, wanx_aug]"
        self.prompts_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{prompt_set_name}_prompt")
        self.eval_dimension_files = [
            os.path.join(self.prompts_dir, file_name) 
            for file_name in os.listdir(self.prompts_dir)
        ]
        self.num_eval_dimensions = len(self.eval_dimension_files)
        # Eval dimension names
        self.eval_dimensions = sorted([file.split(".")[0] for file in os.listdir(self.prompts_dir) if file.endswith(".txt")])
        # Load prompts
        self.eval_prompts = []  # [(prompt, eval_dimension, sample_name)]
        self.meta_info_all = []  # [{"prompt_en": ..., "dimension": ..., "auxiliary_info": ..." }], for VBench-2.0 evaluation only
        for dimension in tqdm(self.eval_dimensions, desc=f"Loading prompts from {self.prompts_dir}"):
            meta_info_file = os.path.join(self.meta_info_dir, dimension + ".json")
            meta_info = json.load(open(meta_info_file, "r"))
            with open(os.path.join(self.prompts_dir, dimension + ".txt"), "r") as f:
                prompts = f.readlines()
                if num_prompts_per_dimension is not None and num_prompts_diversity is not None:
                    # Only use a subset of prompts if num_prompts_per_dimension is specified
                    # The "Diversity" dimension has a different number of prompts, 
                    # since each "Diverisity" prompt requires at least 20 videos for evaluation.
                    num_prompts = num_prompts_per_dimension if dimension != "Diversity" else num_prompts_diversity
                    interval = len(prompts) // num_prompts
                    idx, mod = [], 0
                    while len(idx) < num_prompts:
                        idx.extend([i for i in range(len(prompts)) if i % interval == mod])
                    idx = sorted(idx)[:num_prompts]
                    prompts = [p for i, p in enumerate(prompts) if i in idx]
                    meta_info = [m for i, m in enumerate(meta_info) if i in idx]
                    for i, info in enumerate(meta_info):
                        info["prompt_en"] = prompts[i]
                        info["dimension"] = [dimension]
                for prompt in prompts:
                    sample_num = num_videos_per_prompt if dimension != "Diversity" else num_videos_per_prompt_diversity
                    for i in range(sample_num):
                        self.eval_prompts.append((prompt.strip(), dimension, f"{prompt.strip()[:180]}-{i}"))
                self.meta_info_all.extend(meta_info)
        json.dump(self.meta_info_all, open(os.path.join(self.prompts_dir, f"{prompt_set_name}_{len(self.eval_prompts)}_info.json"), "w"))

    def __len__(self):
        return len(self.eval_prompts)

    def __getitem__(self, idx):
        return self.eval_prompts[idx]
