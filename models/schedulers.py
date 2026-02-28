import numpy as np
from typing import List, Optional, Union

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler
)

"""class FlowMatchEulerDiscreteInverseScheduler(FlowMatchEulerDiscreteScheduler):
    
    def set_timesteps(
        self,
        num_inference_steps: int = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
    ):

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(" you have a pass a value for `mu` when `use_dynamic_shifting` is set to be `True`")

        if sigmas is None:
            self.num_inference_steps = num_inference_steps
            timesteps = np.linspace(
                self._sigma_to_t(self.sigma_max), self._sigma_to_t(self.sigma_min), num_inference_steps
            )

            sigmas = timesteps / self.config.num_train_timesteps

        if self.config.use_dynamic_shifting:
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device=device)
        sigmas = torch.flip(sigmas, [0])
        timesteps = sigmas * self.config.num_train_timesteps

        self.timesteps = timesteps.to(device=device)
        self.sigmas = torch.cat([torch.zeros(1, device=sigmas.device), sigmas])

        self._step_index = None
        self._begin_index = None"""


class FlowMatchEulerDiscreteInverseScheduler(FlowMatchEulerDiscreteScheduler):
    
    _compatibles = ["FlowMatchEulerDiscreteScheduler"]  # 便于 from_config 兼容

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        sigmas: Optional[List[float]] = None,
        mu: Optional[float] = None,
        timesteps: Optional[List[float]] = None,
    ):
        
        super().set_timesteps(
            num_inference_steps=num_inference_steps,
            device=device,
            sigmas=sigmas,
            mu=mu,
            timesteps=timesteps,
        )
        
        self.timesteps = torch.flip(self.timesteps, dims=[0]).contiguous()
        self.sigmas = torch.flip(self.sigmas, dims=[0]).contiguous()
        
        self._step_index = None
        self._begin_index = None
