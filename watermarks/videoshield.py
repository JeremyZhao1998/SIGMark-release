import torch
import numpy as np
import pickle as pkl
from diffusers.utils.torch_utils import randn_tensor
from scipy.stats import norm, truncnorm
from Crypto.Cipher import ChaCha20
from Crypto.Random import get_random_bytes

from watermarks.video_gen_watermark_base import VideoGenWatermarkBase
from utils.distributed_utils import is_dist_avail_and_initialized, all_gather, get_rank


class VideoShieldWatermark(VideoGenWatermarkBase):

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
        VideoShield watermarking, proposed by:
        Videoshield: Regulating diffusion-based video generation models via watermarking, ICLR 2025
        Runyi Hu, et al. (NTU)

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
        self.maintained_info = []
        self.encrypted_messages, self.keys, self.nonces = None, None, None
        if self.maintained_info_path is not None:
            self.load_maintained_info(self.maintained_info_path)
    
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
        repeated_message = np.tile(watermark_message, (1, self.ch_factor, self.fr_factor, self.hw_factor, self.hw_factor))
        # repeated_message_flattened: (batch_size, latent_c * latent_f * latent_h * latent_w)
        repeated_message_flattened = repeated_message.reshape(self.batch_size, -1)
        # encrypted_message_flattened: (batch_size, latent_c * latent_f * latent_h * latent_w)
        encrypted_message_flattened, keys, nonces = self.stream_key_encrypt(repeated_message_flattened)
        # watermarked_latents_flattened: (batch_size, latent_c * latent_f * latent_h * latent_w)
        watermarked_latents_flattened = self.trunc_sampling(encrypted_message_flattened)
        # watermarked_latents: (batch_size, latent_c, latent_f, latent_h, latent_w)
        watermarked_latents = watermarked_latents_flattened.reshape(*self.get_latent_shape())
        watermarked_latents = torch.as_tensor(watermarked_latents, dtype=self.dtype, device=self.device)
        first_frame_latent_shape = (self.batch_size, self.latent_c, 1, self.latent_h, self.latent_w)
        first_frame_latent = randn_tensor(first_frame_latent_shape, device=self.device, dtype=self.dtype)
        latents_all = torch.cat((first_frame_latent, watermarked_latents), dim=2)
        # Accumulate watermark info
        if is_dist_avail_and_initialized():
            watermark_message_reduced = all_gather(watermark_message)
            watermark_message = np.concatenate(watermark_message_reduced, axis=0)
            keys_reduced = all_gather(keys)
            keys = []
            for key in keys_reduced:
                keys.extend(key)
            nonces_reduced = all_gather(nonces)
            nonces = []
            for nonce in nonces_reduced:
                nonces.extend(nonce)
        for (m, k, n) in zip(watermark_message, keys, nonces):
            self.maintained_info.append((m, k, n))
        return latents_all
    
    def match_key_nonce(self, latent):
        """
        Match the correct key and nonce for a given latent.

        Parameters
        ----------
        latent: The inversed latent noise of one video, including the first frame
            torch.Tensor, shape (latent_c, (latent_f + 1), latent_h, latent_w)

        Returns
        -------
        key: The key used to encrypt the latent
        nonce: The nonce used to encrypt the latent
        reordered_latent: The reordered latent, including the first frame
            
        """
        marked_latent = latent[:, 1:, :, :]  # shape (latent_c, latent_f, latent_h, latent_w)
        predicted_message = (marked_latent > 0).to(torch.uint8).transpose(0, 1)  # shape (latent_f, latent_c, latent_h, latent_w)
        predicted_message_flt = predicted_message.reshape(self.latent_f, -1)  # shape (latent_f, latent_c * latent_h * latent_w)
        assert self.encrypted_messages is not None, "Please load maintained info first"
        sample_num = self.encrypted_messages.shape[0]
        encrypted_messages_all = self.encrypted_messages.transpose(1, 2).contiguous()
        encrypted_messages_all_flt = encrypted_messages_all.view(sample_num, self.latent_f, -1) 
        frame_sims = torch.eq(predicted_message_flt.unsqueeze(0), encrypted_messages_all_flt.unsqueeze(2))
        frame_sims = frame_sims.to(self.dtype).mean(dim=-1)
        frame_orders = frame_sims.argmax(dim=-1)
        sample_sims = frame_sims.max(dim=-1).values.mean(dim=-1)
        sample_idx = sample_sims.argmax().item()
        key = self.keys[sample_idx]
        nonce = self.nonces[sample_idx]
        frame_order_idx = frame_orders[sample_idx]
        reordered_marked_latent = marked_latent[:, frame_order_idx, :, :]
        reordered_latent = torch.cat((latent[:, 0:1, :, :], reordered_marked_latent), dim=1)
        return key, nonce, reordered_latent

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
        keys, nonces, reordered_latents = [], [], []
        for latent in latent_all.cpu():
            k, n, l = self.match_key_nonce(latent)
            keys.append(k)
            nonces.append(n)
            reordered_latents.append(l)
        reordered_latent_all = torch.stack(reordered_latents, dim=0)
        watermarked_latents = reordered_latent_all[:, :, 1:, :, :]  # the first frame is not watermarked
        # encrypted_message_flattened: (batch_size, latent_c * latent_f * latent_h * latent_w)
        encrypted_message_flattened = (watermarked_latents > 0).int().view(self.batch_size, -1)
        encrypted_message_flattened = encrypted_message_flattened.cpu().numpy()
        # repeated_message_flattened: (batch_size, latent_c * latent_f * latent_h * latent_w)
        repeated_message_flattened = self.stream_key_decrypt(encrypted_message_flattened, keys, nonces)
        # watermark_message: (batch_size, watermark_c, watermark_f, watermark_h, watermark_w)
        extracted_watermark_message = self.watermark_voting(repeated_message_flattened)
        return extracted_watermark_message

    def stream_key_encrypt(self, message, keys=None, nonces=None):
        """
        Encrypts the given message using ChaCha20.

        Parameters
        ----------
        message : numpy.ndarray, shape (batch_size, latent_len), range {0, 1}
            The input message to be encrypted.
        keys : list, len: batch_size, if None, random keys will be generated
            The encryption keys.
        nonces : list, len: batch_size, if None, random nonces will be generated
            The nonces.

        Returns
        -------
        encrypted_message : numpy.ndarray, shape (batch_size, latent_len), dtype np.uint8
            The encrypted message.
        keys : list, len: batch_size
            The encryption keys.
        nonces : list, len: batch_size
            The nonces.
        """
        encrypted_message = []
        if keys is None or nonces is None:
            keys, nonces = [], []
            for m in message:
                if self.seed is not None:
                    k = np.random.randint(0, 256, 32, dtype=np.uint8).tobytes()
                    n = np.random.randint(0, 256, 12, dtype=np.uint8).tobytes()
                else:
                    k = get_random_bytes(32)
                    n = get_random_bytes(12)
                keys.append(k)
                nonces.append(n)
        else:
            keys = [k.tobytes() for k in keys]
            nonces = [n.tobytes() for n in nonces]
        for m, k, n in zip(message, keys, nonces):
            cipher = ChaCha20.new(key=k, nonce=n)
            encrepted_m_byte = cipher.encrypt(np.packbits(m).tobytes())
            encrypted_m = np.unpackbits(np.frombuffer(encrepted_m_byte, dtype=np.uint8))
            encrypted_message.append(encrypted_m)
        encrypted_message = np.stack(encrypted_message)
        keys = [np.frombuffer(k, dtype=np.uint8) for k in keys]
        nonces = [np.frombuffer(n, dtype=np.uint8) for n in nonces]
        return encrypted_message, keys, nonces
    
    def stream_key_decrypt(self, encrypted_message, keys=None, nonces=None):
        """
        Decrypts the given encrypted message using ChaCha20.

        Parameters
        ----------
        encrypted_message : numpy.ndarray, shape (batch_size, latent_len), dtype np.uint8
            The encrypted message to be decrypted.
        keys : list, len: batch_size
            The encryption keys.
        nonces: list, len: batch_size
            The nonces.

        Returns
        -------
        messages : numpy.ndarray, shape (batch_size, latent_len), range {0, 1}
            The decrypted message. 
        """
        messages = []
        for em, k, n in zip(encrypted_message, keys, nonces):
            cipher = ChaCha20.new(key=k.tobytes(), nonce=n.tobytes())
            message_byte = cipher.decrypt(np.packbits(em).tobytes())
            message = np.unpackbits(np.frombuffer(message_byte, dtype=np.uint8))
            messages.append(message)
        messages = np.stack(messages)
        return messages
    
    @staticmethod
    def trunc_sampling(message):
        """
        Fast implementation of the truncated sampling.
        When random seed is fixed, the output is deterministic.

        Parameters
        ----------
        message : numpy.ndarray, shape (batch_size, latent_len), range {0, 1}
            The input message to be sampled.

        Returns
        -------
        z : numpy.ndarray, shape (batch_size, latent_len), dtype self.dtype
            The sampled latent.
        """
        batch_size, latent_len = message.shape
        # precompute bounds
        neg_inf, zero, pos_inf = norm.ppf([0, 0.5, 1])
        bounds = np.array([[neg_inf, zero], [zero, pos_inf]])  # shape (2,2)
        # sample
        a, b = bounds[message.flatten()].T  # shape (2, latent_len)
        z = truncnorm.rvs(a, b)
        return z.reshape(batch_size, latent_len)
    
    def watermark_voting(self, repeated_message):
        """
        Voting to extract the watermark message from the repeated message.
        For each repeated position, the mean value is used to determine whether it is 0 or 1.

        Parameters
        ----------
        repeated_message : The repeated message to be voted.
            numpy.ndarray, shape (batch_size, latent_c * latent_f * latent_h * latent_w), range {0, 1}
            
        Returns
        -------
        voted_message : The voted watermark message.
            numpy.ndarray, shape (batch_size, watermark_c, watermark_f, watermark_h, watermark_w)
            
        """
        reshaped_repeated_message = repeated_message.reshape(
            self.batch_size,
            self.ch_factor, self.watermark_c,
            self.fr_factor, self.watermark_f,
            self.hw_factor, self.watermark_h,
            self.hw_factor, self.watermark_w
        ).astype(np.float32)
        scores = np.mean(reshaped_repeated_message, axis=(1, 3, 5, 7))
        voted_message = (scores >= 0.5).astype(np.uint8)
        return voted_message
    
    def save_maintained_info(self, filename):
        assert filename.endswith(".pkl")
        if not is_dist_avail_and_initialized() or get_rank() == 0:
            pkl.dump(self.maintained_info, open(filename, "wb"))

    def load_maintained_info(self, filename):
        assert filename.endswith(".pkl")
        print("loading maintained info from {}".format(filename))
        self.maintained_info = np.load(filename, allow_pickle=True)
        messages_all, self.keys, self.nonces = [], [], []
        for (m, k, n) in self.maintained_info:
            messages_all.append(m)
            self.keys.append(k)
            self.nonces.append(n)
        messages_all = np.stack(messages_all)
        repeated_messages_all = np.tile(messages_all, (1, self.ch_factor, self.fr_factor, self.hw_factor, self.hw_factor))
        repeated_messages_all_flt = repeated_messages_all.reshape(messages_all.shape[0], -1)
        encrypted_messages_all_flt, _, _ = self.stream_key_encrypt(repeated_messages_all_flt, self.keys, self.nonces)
        encrypted_messages_all_flt = torch.from_numpy(encrypted_messages_all_flt).detach().to("cpu")
        encrypted_messages_all = encrypted_messages_all_flt.view(messages_all.shape[0], self.latent_c, self.latent_f, 
                                                                 self.latent_h, self.latent_w)
        self.encrypted_messages = encrypted_messages_all
