import torch
import cv2
import random
import multiprocessing
import numpy as np
import pickle as pkl
from PIL import Image
from scipy.special import erf
from tqdm import tqdm
from diffusers.utils.torch_utils import randn_tensor

from watermarks.video_gen_watermark_base import VideoGenWatermarkBase
from utils.prc_utils import (
    pseudorandom_code_key_gen,
    pseudorandom_code_encode,
    pseudorandom_code_detect,
    pseudorandom_code_decode,
)
from utils.distributed_utils import (
    is_dist_avail_and_initialized, 
    get_rank, 
    all_gather
)
from utils.video_utils import (
    align_frames,
    pad_video_by_interpolation,
    optical_flow_segmentation,
    pad_video_by_interpolation
)


def _pseudorandom_code_key_gen_meta(args):
    idx, codeword_len, message_len, false_positive_rate, t, g, r, noise_rate, max_bp_iter = args
    encoding_key, decoding_key = pseudorandom_code_key_gen(codeword_len, message_len, \
        false_positive_rate, t, g, r, noise_rate, max_bp_iter)
    return idx, encoding_key, decoding_key


def _pseudorandom_code_decode_meta(args):
    idx, posteriors, decoding_key = args
    return idx, pseudorandom_code_decode(posteriors, decoding_key)


class SIGMarkWatermark(VideoGenWatermarkBase):

    def __init__(
        self, 
        video_h,
        video_w,
        video_f,
        latent_c,
        vae_scale_factor_spatial,
        vae_scale_factor_temporal,
        ch_factor, 
        hw_factor, 
        fr_factor,
        maintained_info_path=None,
        batch_size=1,
        seed=None,
        dtype=torch.bfloat16,
        device='cuda'
    ):
        super().__init__(video_h, video_w, video_f, latent_c, vae_scale_factor_spatial, vae_scale_factor_temporal, 
                         ch_factor, hw_factor, fr_factor, maintained_info_path, batch_size, seed, dtype, device)
        codeword_len = self.latent_c * self.latent_h * self.latent_w
        message_length = self.watermark_c * self.watermark_h * self.watermark_w
        self.encoding_keys, self.decoding_keys = [None] * self.latent_f, [None] * self.latent_f
        if not is_dist_avail_and_initialized() or get_rank() == 0:
            if self.maintained_info_path is None:
                self._get_keys(codeword_len, message_length)
            else:
                self.load_maintained_info(self.maintained_info_path)
        if is_dist_avail_and_initialized():
            for i in range(self.latent_f):
                self.encoding_keys[i] = all_gather(self.encoding_keys[i])[0]
                self.decoding_keys[i] = all_gather(self.decoding_keys[i])[0]
        self.maintained_info = {
            "encoding_keys": self.encoding_keys,
            "decoding_keys": self.decoding_keys
        }

    def _get_keys(self, codeword_len, message_length):
        with multiprocessing.Pool(
            processes=min(self.latent_f // 2, multiprocessing.cpu_count())
        ) as pool:
            for i_res, encoding_key, decoding_key in pool.imap_unordered(
                _pseudorandom_code_key_gen_meta, 
                [(i, codeword_len, message_length, 0.5, 3, None, None, 0.0, 200) for i in range(self.latent_f)]
            ):
                self.encoding_keys[i_res] = encoding_key
                self.decoding_keys[i_res] = decoding_key

    def create_watermarked_latents(self, watermark_message):
        """
        Generate watermarked latents for video generation.

        Args:
            watermark_message: The watermark message to embed in the latents.
                numpy.ndarray, shape (batch_size, watermark_c, watermark_f, watermark_h, watermark_w)

        Returns:
            latents_all: watermarked latents, including the random sampled first frame latent
                torch.Tensor, shape (batch_size, latent_c, (latent_f + 1), latent_h, latent_w)
        """
        assert list(watermark_message.shape) == self.get_watermark_message_shape()
        # watarmark_message: (batch_size, watermark_c, watermark_f, watermark_h, watermark_w)
        # repeated_message: (batch_size, latent_c, latent_f, latent_h, latent_w)
        repeated_message = np.tile(watermark_message, (1, 1, self.fr_factor, 1, 1))
        repeated_message_flt = repeated_message.transpose((0, 2, 1, 3, 4)).reshape(self.batch_size, self.latent_f, -1)
        prc_codewords = [[] for _ in range(self.batch_size)]
        for b in range(self.batch_size):
            for f in range(self.latent_f):
                prc_codeword = pseudorandom_code_encode(repeated_message_flt[b][f], self.encoding_keys[f])
                prc_codewords[b].append(prc_codeword)
        prc_codewords = np.array(prc_codewords)
        prc_codewords = prc_codewords.reshape((self.batch_size, self.latent_f, self.latent_c, self.latent_h, self.latent_w))
        prc_codewords = prc_codewords.transpose((0, 2, 1, 3, 4))
        # latents: (batch_size, latent_c, latent_f, latent_h, latent_w)
        latents = randn_tensor(self.get_latent_shape(), device=self.device, dtype=self.dtype)
        # watermarked_latents: (batch_size, latent_c, latent_f, latent_h, latent_w)
        watermarked_latents = torch.abs(latents) * torch.as_tensor(prc_codewords, dtype=self.dtype, device=self.device)
        first_frame_latent_shape = (self.batch_size, self.latent_c, 1, self.latent_h, self.latent_w)
        first_frame_latent = randn_tensor(first_frame_latent_shape, device=self.device, dtype=self.dtype)
        latents_all = torch.cat((first_frame_latent, watermarked_latents), dim=2)
        return latents_all

    @staticmethod
    def _recover_posteriors(latents, basis=None, variances=1.5):
        denominators = np.sqrt(2 * variances * (1 + variances))
        if basis is None:
            return erf((latents / denominators))
        else:
            return erf(((latents @ basis) / denominators))
        
    def detect_frame_idx(self, watermarked_latents):
        watermarked_latents_np = watermarked_latents.cpu().float().numpy()
        batch_size, latent_c, latent_f, latent_h, latent_w = watermarked_latents_np.shape
        watermarked_latents_flt = watermarked_latents_np.transpose((0, 2, 1, 3, 4)).reshape(batch_size * latent_f, -1)
        posteriors = self._recover_posteriors(watermarked_latents_flt)
        detection_scores = [[0.0 for _ in range(len(self.decoding_keys))] for _ in range(posteriors.shape[0])]
        for p_idx, p in enumerate(posteriors):
            for k_idx, k_set in enumerate(self.decoding_keys):
                for i in range(4):
                    idx_start, idx_end = i * p.shape[0] // 4, (i + 1) * p.shape[0] // 4
                    detection_scores[p_idx][k_idx] += pseudorandom_code_detect(p[idx_start: idx_end], k_set[i])[1]
        detection_results = np.argmax(np.array(detection_scores), axis=1)
        detected_frame_idx = detection_results.reshape((batch_size, latent_f))
        return detected_frame_idx
    
    def extract_watermark(self, latent_all):
        """
        Extract the watermark from the latent space of the input video.

        Parameters
        ----------
        latent_all: The inversed latent noise of the video, including the first frame
            torch.Tensor, shape (batch_size, latent_c, (latent_f + 1), latent_h, latent_w)

        Returns
        -------
        extracted_watermark_message: The extracted watermark message.
            
        """
        assert len(latent_all.shape) == 5 and latent_all.shape[0] == self.batch_size
        watermarked_latents = latent_all[:, :, 1:, :, :]  # the first frame is not watermarked
        watermarked_latents_np = watermarked_latents.cpu().float().numpy()
        watermarked_latents_flt = watermarked_latents_np.transpose((0, 2, 1, 3, 4)).reshape(self.batch_size * self.latent_f, -1)
        posteriors = self._recover_posteriors(watermarked_latents_flt, variances=1.5)
        detection_results = [[pseudorandom_code_detect(p, key)[1] for key in self.decoding_keys] for p in posteriors]
        key_idx = np.argmax(np.array(detection_results), axis=1)
        decoding_keys = [self.decoding_keys[i] for i in key_idx]
        decoded_messages = [None] * len(posteriors)
        with multiprocessing.Pool(processes=min(self.latent_f // 2, multiprocessing.cpu_count())) as pool:
            for i_res, msg in tqdm(
                pool.imap_unordered(_pseudorandom_code_decode_meta, [(i, p, decoding_keys[i]) for i, p in enumerate(posteriors)]), 
                total=len(posteriors)
            ):
                decoded_messages[i_res] = msg.astype(np.float32)
        decoded_messages = np.array(decoded_messages).reshape((self.batch_size, self.latent_f, -1))
        reordered_messages = np.ones_like(decoded_messages) * -1
        reordered_idx = key_idx.reshape((self.batch_size, self.latent_f))
        for b in range(self.batch_size):
            for f in range(self.latent_f):
                reordered_messages[b, f, :] = decoded_messages[b, reordered_idx[b, f], :]
        decoded_messages = reordered_messages.reshape((self.batch_size, self.fr_factor, self.watermark_f, -1))
        voted_decoded_messages = (np.mean(decoded_messages, axis=1) >= 0.5).astype(np.uint8)
        predicted_messages = voted_decoded_messages.reshape(self.batch_size, self.watermark_f, self.watermark_c, self.watermark_h, self.watermark_w)
        predicted_messages = predicted_messages.transpose((0, 2, 1, 3, 4))
        return predicted_messages

    def save_maintained_info(self, filename):
        if not is_dist_avail_and_initialized() or get_rank() == 0:
            assert filename.endswith(".pkl")
            with open(filename, "wb") as f:
                pkl.dump(self.maintained_info, f)

    def load_maintained_info(self, filename):
        print("loading maintained info from {}".format(filename))
        assert filename.endswith(".pkl")
        with open(filename, "rb") as f:
            self.maintained_info = pkl.load(f)
        self.encoding_keys = self.maintained_info["encoding_keys"]
        self.decoding_keys = self.maintained_info["decoding_keys"]

    def _get_single_latent(self, video_frames, pipeline):
        frame_tensors = pipeline.video_processor.preprocess_video(video_frames)
        latents = pipeline.vae.encode(frame_tensors.to(pipeline.vae.dtype).to(self.device)).latent_dist.mode()
        latents *= pipeline.vae.config.scaling_factor
        inverted_init_latents = []
        for i in range(len(video_frames)):
            inverted_init_latent = pipeline(
                prompt="",
                guidance_scale=6.0,
                num_inference_steps=50,
                num_frames=len(video_frames[i]),
                latents=latents[i].unsqueeze(0),
                output_type="latent",
                return_dict=False,
                image=video_frames[i][0],
                height=video_frames[i][0].height,
                width=video_frames[i][0].width
            )[0]
            inverted_init_latents.append(inverted_init_latent)
        inverted_init_latents = torch.cat(inverted_init_latents, dim=0)
        marked_mini_batch_latents = inverted_init_latents[:, :, 1:, :, :].cpu().float().numpy()
        marked_mini_batch_latents = marked_mini_batch_latents.transpose((0, 2, 1, 3, 4))
        batch_size, latent_f = marked_mini_batch_latents.shape[0], marked_mini_batch_latents.shape[1]
        marked_mini_batch_latents = marked_mini_batch_latents.reshape((batch_size, latent_f, -1))
        return marked_mini_batch_latents
    
    def _get_prc_detection_results(self, mini_batch_latents):
        posteriors = self._recover_posteriors(mini_batch_latents)
        results = np.array([
            [[pseudorandom_code_detect(p, key)[1] for key in self.decoding_keys] for p in posterior] 
            for posterior in posteriors
        ])
        key_idx = np.argmax(results, axis=2)
        key_score = np.max(results, axis=2).sum(axis=-1)
        qualified_key_idx = [i for i in range(len(key_idx)) if key_idx[i][0] + 1 == key_idx[i][1]]
        if len(qualified_key_idx) == 0:
            return -1, -1
        if len(qualified_key_idx) == 1:
            return qualified_key_idx[0], key_idx[qualified_key_idx[0]][0]
        else:
            chosen_key_idx = qualified_key_idx[np.argmax([key_score[i] for i in qualified_key_idx])]
            return chosen_key_idx, key_idx[chosen_key_idx][0]
        
    def segment_group_ordering(self, video_frames, tgt_num_frames, of_seg=True, sw_det=True, pipeline=None):
        first_frame, frames_marked = video_frames[0], video_frames[1:]
        if of_seg:
            segments_idx = optical_flow_segmentation(frames_marked)
        else:
            segments_idx = [(0, len(frames_marked) - 1)]
        if sw_det:
            assert pipeline is not None, "pipeline should be provided when using sw_det"
            reordered_frames = [None] * (tgt_num_frames - 1)
            group_frames = pipeline.vae_scale_factor_temporal
            for start, end in segments_idx:
                seg_frames = frames_marked[start: end + 1]
                if len(seg_frames) <= group_frames:
                    padded_seg_frames = [seg_frames[0]] + seg_frames + [seg_frames[-1]] * (group_frames - len(seg_frames))
                    latents = self._get_single_latent([padded_seg_frames], pipeline)
                    posteriors = self._recover_posteriors(latents)
                    result = [pseudorandom_code_detect(posteriors[0][0], key)[1] for key in self.decoding_keys]
                    key_idx = np.argmax(result)
                    reordered_frames[key_idx * group_frames: key_idx * group_frames + len(seg_frames)] = seg_frames
                else:
                    padded_seg_frames = [seg_frames[0]] * (group_frames - 1) + seg_frames + [seg_frames[-1]] * (group_frames - 1)
                    mini_batch = []
                    for i in range(group_frames):
                        mini_batch.append([padded_seg_frames[i]] + padded_seg_frames[i: i + group_frames * 2])
                    mini_batch_latents = self._get_single_latent(mini_batch, pipeline)
                    sliding_window_start, latent_f_idx = self._get_prc_detection_results(mini_batch_latents)
                    if sliding_window_start == -1 or latent_f_idx == -1:
                        continue
                    chosen_seg_frames = padded_seg_frames[sliding_window_start: -(group_frames - 1)]
                    f_idx = latent_f_idx * group_frames
                    reordered_frames[f_idx: f_idx + len(chosen_seg_frames)] = chosen_seg_frames
            reordered_frames = pad_video_by_interpolation(reordered_frames)
        else:
            if len(frames_marked) > tgt_num_frames - 1:
                seg_lens = [end - start + 1 for start, end in segments_idx]
                most_common_len = max(set(seg_lens), key=seg_lens.count)
                reordered_frames = []
                for start, end in segments_idx:
                    if end - start + 1 != most_common_len:
                        reordered_frames.extend(frames_marked[start: end + 1])
                reordered_frames = align_frames(reordered_frames, tgt_num_frames - 1)
            elif len(frames_marked) < tgt_num_frames - 1:
                total_pad = tgt_num_frames - 1 - len(frames_marked)
                num_segments = len(segments_idx)
                if num_segments == 1:
                    reordered_frames = [None] * (tgt_num_frames - 1)
                    for idx in range(total_pad // 2, total_pad // 2 + len(frames_marked)):
                        reordered_frames[idx] = frames_marked[idx - total_pad // 2]
                else:
                    base, rem = total_pad // num_segments, total_pad % num_segments
                    gap_pads = [base + 1 if i < rem else base for i in range(num_segments)]
                    reordered_frames = []
                    for i, (start, end) in enumerate(segments_idx):
                        reordered_frames.extend([None] * gap_pads[i])
                        reordered_frames.extend(frames_marked[start: end + 1])
            else:
                reordered_frames = frames_marked
        reordered_frames = [first_frame] + reordered_frames
        reordered_frames = pad_video_by_interpolation(reordered_frames)
        return reordered_frames
