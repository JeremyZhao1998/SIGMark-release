import torch
import multiprocessing
import numpy as np
import pickle as pkl
from scipy.special import erf
from Levenshtein import distance
from diffusers.utils.torch_utils import randn_tensor
from tqdm import tqdm

from watermarks.video_gen_watermark_base import VideoGenWatermarkBase
from utils.distributed_utils import (
    is_dist_avail_and_initialized, 
    get_rank, 
    all_gather
)
from utils.prc_utils import (
    pseudorandom_code_key_gen,
    pseudorandom_code_encode,
    pseudorandom_code_detect,
    pseudorandom_code_decode
)


def _init_worker(decoding_key):
    global DEC_KEY
    DEC_KEY = decoding_key


def _pseudorandom_code_decode_meta(args):
    idx, posteriors = args
    return idx, pseudorandom_code_decode(posteriors, DEC_KEY)


class VideoMarkWatermark(VideoGenWatermarkBase):
    
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
        """
        VideoMark watermarking, proposed by:
        VideoMark: A Distortion-Free Robust Watermarking Framework for Video Diffusion Models
        Xuming Hu, Hanqian Li, Jungang Li, et al. (HUKST)

        We re-implement some of the functions from the original code to improve flexibility and inference speed.

        Args:
            video_h (int): height of the generated video
            video_w (int): width of the generated video
            video_f (int): number of frames in the generated video
            latent_c (int): number of channels in the latent space
            vae_scale_factor_spatial (int): scale factor for spatial dimensions of the latent space
            vae_scale_factor_temporal (int): scale factor for temporal dimension of the latent space
            ch_factor (int): watermark message repeated along the channel dimension
            hw_factor (int): watermark message repeated along the height and width dimensions
            fr_factor (int): watermark message repeated along the frame dimension
            batch_size (int, optional): batch size for the watermark embeddings. Defaults to 1.
            seed (int, optional): seed for the random number generator. Defaults to None.
            dtype (torch.dtype, optional): dtype of video generation. Defaults to torch.bfloat16.
            device (str, optional): compute device. Defaults to 'cuda'.
        """
        super().__init__(video_h, video_w, video_f, latent_c, vae_scale_factor_spatial, vae_scale_factor_temporal, 
                         ch_factor, hw_factor, fr_factor, maintained_info_path, batch_size, seed, dtype, device)
        codeword_len = self.latent_c * self.latent_h * self.latent_w
        message_length = self.watermark_c * self.watermark_h * self.watermark_w
        if not is_dist_avail_and_initialized() or get_rank() == 0:
            if self.maintained_info_path is None:
                self.encoding_key, self.decoding_key = pseudorandom_code_key_gen(codeword_len, message_length)
                self.maintained_info = {
                    "watermark_messages": [],
                    "encoding_keys": self.encoding_key,
                    "decoding_keys": self.decoding_key
                }
            else:
                self.load_maintained_info(self.maintained_info_path)
        if is_dist_avail_and_initialized():
            self.encoding_key = all_gather(self.encoding_key)[0]
            self.decoding_key = all_gather(self.decoding_key)[0]
        
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
        # repeated_message_flattened: (batch_size, latent_c * latent_f * latent_h * latent_w)
        repeated_message_flt = repeated_message.transpose((0, 2, 1, 3, 4)).reshape(self.batch_size, self.latent_f, -1)
        prc_codewords = [[] for _ in range(self.batch_size)]
        for b in range(self.batch_size):
            for f in range(self.latent_f):
                prc_codeword = pseudorandom_code_encode(repeated_message_flt[b][f], self.encoding_key)
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
        # Accumulate watermark info
        if is_dist_avail_and_initialized():
            watermark_message_reduced = all_gather(watermark_message)
            watermark_message = np.concatenate(watermark_message_reduced, axis=0)
        for m in watermark_message:
            self.maintained_info["watermark_messages"].append(m)
        return latents_all
    
    @staticmethod
    def _recover_posteriors(latents, basis=None, variances=1.5):
        denominators = np.sqrt(2 * variances * (1 + variances))
        if basis is None:
            return erf((latents / denominators))
        else:
            return erf(((latents @ basis) / denominators))
        
    @staticmethod
    def _bit_to_str(bit):
        return "".join(map(str, bit))
    
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
        detection_results = [pseudorandom_code_detect(p, self.decoding_key)[0] for p in posteriors]
        decoded_messages = [None] * len(posteriors)
        with multiprocessing.Pool(
            processes=min(self.latent_f // 2, multiprocessing.cpu_count()),
            initializer=_init_worker,
            initargs=(self.decoding_key,)
        ) as pool:
            for i_res, msg in tqdm(
                pool.imap_unordered(_pseudorandom_code_decode_meta, [(i, p) for i, p in enumerate(posteriors)]), 
                total=len(posteriors)
            ):
                decoded_messages[i_res] = msg.astype(np.float32) if detection_results[i_res] \
                    else np.ones_like(msg).astype(np.float32) * 0.5
        decoded_messages = np.array(decoded_messages).reshape((self.batch_size, self.fr_factor, self.watermark_f, -1))
        voted_decoded_messages = (np.mean(decoded_messages, axis=1) >= 0.5).astype(np.uint8).reshape(self.batch_size * self.watermark_f, -1)
        watermark_messages = np.stack(self.maintained_info["watermark_messages"]).transpose((0, 2, 1, 3, 4))
        total_frame_num = watermark_messages.shape[0]
        watermark_messages = watermark_messages.reshape(total_frame_num * self.watermark_f, -1)
        predicted_messages = []
        for msg in voted_decoded_messages:
            d_list = np.array([distance(msg, gt_msg) for gt_msg in watermark_messages])
            msg_dix = np.argmin(d_list)
            predicted_messages.append(watermark_messages[msg_dix])
        predicted_messages = np.array(predicted_messages)
        predicted_messages = predicted_messages.reshape(self.batch_size, self.watermark_f, self.watermark_c, self.watermark_h, self.watermark_w)
        predicted_messages = predicted_messages.transpose((0, 2, 1, 3, 4))
        return predicted_messages

    def save_maintained_info(self, filename):
        assert filename.endswith(".pkl")
        if not is_dist_avail_and_initialized() or get_rank() == 0:
            with open(filename, "wb") as f:
                pkl.dump(self.maintained_info, f)

    def load_maintained_info(self, filename):
        print("loading maintained info from {}".format(filename))
        assert filename.endswith(".pkl")
        with open(filename, "rb") as f:
            self.maintained_info = pkl.load(f)
        self.encoding_key = self.maintained_info["encoding_key"]
        self.decoding_key = self.maintained_info["decoding_key"]
