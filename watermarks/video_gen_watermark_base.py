import torch
from diffusers.utils.torch_utils import randn_tensor
import numpy as np


class VideoGenWatermarkBase:

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
        device='cuda:0'
    ):
        """
        Base class of in-generation video watermark embedding and extraction for diffusion-based 
        video generation models.
        This class provides no watermark embedding or extraction methods.

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
            maintained_info_path (str, optional): path to the maintained info file. Defaults to None.
            batch_size (int, optional): batch size for the watermark embeddings. Defaults to 1.
            seed (int, optional): seed for the random number generator. Defaults to None.
            dtype (torch.dtype, optional): dtype of video generation. Defaults to torch.bfloat16.
            device (str, optional): compute device. Defaults to 'cuda'.

        Notes:
            - The watermark is embedded in the latent space of the video, 
              so the spatial and temporal dimensions of the watermark must be divisible 
              by the corresponding factors.
            - The first frame of the video is not watermarked, because the first frame is
              often processed independently by the video generation model, saved for I2V generation.
        """
        assert video_h % vae_scale_factor_spatial == 0 and video_w % vae_scale_factor_spatial == 0, \
            "video_h and video_w must be divisible by vae_scale_factor_spatial"
        assert (video_f - 1) % vae_scale_factor_temporal == 0, \
            "(video_f - 1) must be divisible by vae_scale_factor_temporal"
        self.batch_size = batch_size
        self.video_h = video_h
        self.video_w = video_w
        self.video_f = video_f
        self.latent_c = latent_c
        self.vae_scale_factor_spatial = vae_scale_factor_spatial
        self.vae_scale_factor_temporal = vae_scale_factor_temporal
        self.latent_h = self.video_h // self.vae_scale_factor_spatial
        self.latent_w = self.video_w // self.vae_scale_factor_spatial
        self.latent_f = (self.video_f - 1) // self.vae_scale_factor_temporal  # not including the first frame
        assert self.latent_h % hw_factor == 0 and self.latent_w % hw_factor == 0, \
            "latent_h and latent_w must be divisible by hw_factor"
        assert self.latent_f % fr_factor == 0, "latent_f must be divisible by fr_factor"
        assert self.latent_c % ch_factor == 0, "num_channels_latents must be divisible by ch_factor"
        self.ch_factor = ch_factor
        self.hw_factor = hw_factor
        self.fr_factor = fr_factor
        self.watermark_h = self.latent_h // self.hw_factor
        self.watermark_w = self.latent_w // self.hw_factor
        self.watermark_f = self.latent_f // self.fr_factor
        self.watermark_c = self.latent_c // self.ch_factor
        self.seed = seed
        self.dtype = dtype
        self.device = device
        # Information maintained by watermarking system
        self.maintained_info_path = maintained_info_path
        self.maintained_info = None

    def get_latent_shape(self):
        """
        Get the shape of the latent space, including the batch size and dimensions.

        Returns
        -------
        list
            A list representing the shape of the latent space as
            [batch_size, latent_c, latent_f, latent_h, latent_w].
        """
        return [self.batch_size, self.latent_c, self.latent_f, self.latent_h, self.latent_w]

    def get_watermark_len(self):
        """
        Calculate the total length of the watermark per video sample.

        Returns
        -------
        int
            The total number of elements in the watermark, computed as the product
            of its channel, frame, height, and width dimensions.
        """
        return self.watermark_c * self.watermark_f * self.watermark_h * self.watermark_w
    
    def get_watermark_message_shape(self):
        """
        Get the shape of the watermark message, including the batch size and dimensions.

        Returns
        -------
        list
            A list representing the shape of the watermark message as
            [batch_size, watermark_c, watermark_f, watermark_h, watermark_w].
        """
        return [self.batch_size, self.watermark_c, self.watermark_f, self.watermark_h, self.watermark_w]
    
    def generate_random_watermark_message(self):
        """
        Generate a random watermark message for the given shape.

        The watermark message is a binary tensor of shape
        [batch_size, watermark_c, watermark_f, watermark_h, watermark_w].

        Returns
        -------
        numpy.ndarray
            A numpy array with the watermark message, where each element is either 0 or 1.
        """
        watermark_shape = self.get_watermark_message_shape()
        watermark_message = np.random.randint(0, 2, size=watermark_shape, dtype=np.uint8)
        return watermark_message
    
    def create_watermarked_latents(self, watermark_message):
        """
        Generate watermarked latents for video generation.
        This function provide random initial latents.
        Watermarked latents should be implemented in child classes.

        Args:
            watermark_message: The watermark message to embed in the latents.
                numpy.ndarray, shape (batch_size, watermark_c, watermark_f, watermark_h, watermark_w)

        Returns:
            latents_all: watermarked latents, including the random sampled first frame latent
                torch.Tensor, shape (batch_size, latent_c, (latent_f + 1), latent_h, latent_w)
        """
        assert list(watermark_message.shape) == self.get_watermark_message_shape()
        latents_all_shape = (self.batch_size, self.latent_c, self.latent_f + 1, self.latent_h, self.latent_w)
        latents_all = randn_tensor(latents_all_shape, device=self.device, dtype=self.dtype)
        return latents_all

    def save_maintained_info(self, filename):
        """
        Save the information maintained by the watermarking system to a file.
        This function should be implemented in child classes.

        Parameters
        ----------
        filename : str
            The name of the file to save the information to.
        """
        return
    
    def load_maintained_info(self, filename):
        """
        Load the information maintained by the watermarking system from a file.
        This function should be implemented in child classes.

        Parameters
        ----------
        filename : str
            The name of the file to load the information from.
        """
        return 
