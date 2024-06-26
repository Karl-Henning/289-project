from typing import Optional, Tuple

import gymnasium as gym
from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
from gymnasium import ObservationWrapper

import numpy as np
import torch
import torch.nn as nn

import cv2 as cv

from env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)
from infrastructure.atari_wrappers import wrap_deepmind
import infrastructure.pytorch_util as ptu


class PreprocessCarRacing(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in [3, 4, 5], f"Bad observation shape: {x.shape}"
        assert x.shape[-3:] == (4, 84, 84), f"Bad observation shape: {x.shape}"
        assert x.dtype == torch.uint8

        return x / 255.0

def crop_image(observation, crop_size):
    top, right, bottom, left = crop_size
    return observation[top:-bottom, left:-right]
    
def edge_detection(observation):
    # edge detection
    observation = cv2.Canny(observation, 100, 200)
    return observation

class CropObservationWrapper(ObservationWrapper):
    def __init__(self, env, crop_size):
        super(CropObservationWrapper, self).__init__(env)
        self.crop_size = crop_size

        # Adjust observation space after cropping
        low = np.zeros((84, 84), dtype=np.uint8)
        high = np.ones((84, 84), dtype=np.uint8) * 255
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.uint8)

    def observation(self, observation):
        return crop_image(observation, self.crop_size)

def car_racing_dqn_config(
    env_name: str,
    exp_name: Optional[str] = None,
    learning_rate: float = 1e-4,
    adam_eps: float = 1e-4,
    total_steps: int = 1000000,
    discount: float = 0.99,
    target_update_period: int = 2000,
    clip_grad_norm: Optional[float] = 10.0,
    use_double_q: bool = False,
    learning_starts: int = 20000,
    batch_size: int = 32,
    **kwargs,
):
    def make_critic(observation_shape: Tuple[int, ...], num_actions: int) -> nn.Module:
        # print(observation_shape) # (4, 84, 84, 3)
        assert observation_shape == (
            4,
            84,
            84,
        ), f"Observation shape: {observation_shape}"

        return nn.Sequential(
            PreprocessCarRacing(),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),  # 3136 hard-coded based on img size + CNN layers
            nn.ReLU(),
            nn.Linear(512, num_actions),
        ).to(ptu.device)

    def make_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=learning_rate, eps=adam_eps)

    def make_lr_schedule(
        optimizer: torch.optim.Optimizer,
    ) -> torch.optim.lr_scheduler._LRScheduler:
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            PiecewiseSchedule(
                [
                    (0, 1),
                    (20000, 1),
                    (total_steps / 2, 5e-1),
                ],
                outside_value=5e-1,
            ).value,
        )

    exploration_schedule = PiecewiseSchedule(
        [
            (0, 1.0),
            (20000, 1),
            (total_steps / 2, 0.01),
        ],
        outside_value=0.01,
    )

    def make_env(render: bool = False):
        env = gym.make(env_name, render_mode="rgb_array" if render else None, continuous=False)

        env = RecordEpisodeStatistics(env)

        # convert rgb to grayscale
        env = GrayScaleObservation(env)

        print("shape before: ", env.observation_space.shape)
        crop_size = (0, 6, 12, 6)
        env = CropObservationWrapper(env, crop_size)
        print("shape after: ", env.observation_space.shape)

        # env = TransformObservation(env, edge_detection)

        env = FrameStack(env, num_stack=4)
        return env


    log_string = "{}_{}_d{}_tu{}_lr{}".format(
        exp_name or "dqn",
        env_name,
        discount,
        target_update_period,
        learning_rate,
    )

    if use_double_q:
        log_string += "_doubleq"

    if clip_grad_norm is not None:
        log_string += f"_clip{clip_grad_norm}"

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_optimizer": make_optimizer,
            "make_lr_schedule": make_lr_schedule,
            "discount": discount,
            "target_update_period": target_update_period,
            "clip_grad_norm": clip_grad_norm,
            "use_double_q": use_double_q,
        },
        "log_name": log_string,
        "exploration_schedule": exploration_schedule,
        "make_env": make_env,
        "total_steps": total_steps,
        "batch_size": batch_size,
        "learning_starts": learning_starts,
        **kwargs,
    }
