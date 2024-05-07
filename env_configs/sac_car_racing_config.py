from typing import Tuple, Optional

import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn

from networks.cnn_policy import CNNPolicy
from networks.state_action_value_critic import StateActionCritic
import infrastructure.pytorch_util as ptu

from gymnasium.wrappers.frame_stack import FrameStack
from gymnasium.wrappers.gray_scale_observation import GrayScaleObservation
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics

from env_configs.schedule import (
    LinearSchedule,
    PiecewiseSchedule,
    ConstantSchedule,
)

class PreprocessCarRacing(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim in [3, 4], f"Bad observation shape: {x.shape}"
        assert x.shape[-3:] == (4, 96, 96), f"Bad observation shape: {x.shape}"
        assert x.dtype == torch.uint8

        return x / 255.0

def sac_car_racing_config(
    env_name: str,
    exp_name: Optional[str] = None,
    hidden_size: int = 128,
    num_layers: int = 3,
    actor_learning_rate: float = 3e-4,
    critic_learning_rate: float = 3e-4,
    total_steps: int = 300000,
    random_steps: int = 5000,
    training_starts: int = 10000,
    batch_size: int = 128,
    replay_buffer_capacity: int = 1000000,
    ep_len: Optional[int] = None,
    discount: float = 0.99,
    use_soft_target_update: bool = False,
    target_update_period: Optional[int] = None,
    soft_target_update_rate: Optional[float] = None,
    # Actor-critic configuration
    actor_gradient_type="reinforce",  # One of "reinforce" or "reparametrize"
    num_actor_samples: int = 1,
    num_critic_updates: int = 1,
    # Settings for multiple critics
    num_critic_networks: int = 1,
    target_critic_backup_type: str = "mean",  # One of "doubleq", "min", or "mean"
    # Soft actor-critic
    backup_entropy: bool = True,
    use_entropy_bonus: bool = True,
    temperature: float = 0.1,
    actor_fixed_std: Optional[float] = None,
    use_tanh: bool = True,
):
    def make_critic(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        return StateActionCritic(
            ob_shape=observation_shape,
            ac_dim=action_dim,
            n_layers=num_layers,
            size=hidden_size,
        )

    def make_actor(observation_shape: Tuple[int, ...], action_dim: int) -> nn.Module:
        if len(observation_shape) != 3:
            raise ValueError("Input observation shape should be in HWC format (height, width, channels)")

        return CNNPolicy(
            ac_dim=action_dim,
            obs_shape=observation_shape,
            discrete=False,
            n_conv_layers=num_layers,  # You can adjust this as needed
            conv_channels=[32, 64, 64],  # Example values, adjust as needed
            conv_kernel_sizes=[8, 4, 3],  # Example values, adjust as needed
            conv_strides=[4, 2, 1],  # Example values, adjust as needed
            n_fc_layers=num_layers,  # You can adjust this as needed
            fc_layer_size=hidden_size,  # You can adjust this as needed
            use_tanh=use_tanh,
            state_dependent_std=False if actor_fixed_std is not None else True,
            fixed_std=actor_fixed_std,
        )

    def make_actor_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=actor_learning_rate)

    def make_critic_optimizer(params: torch.nn.ParameterList) -> torch.optim.Optimizer:
        return torch.optim.Adam(params, lr=critic_learning_rate)

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

    def make_env(render: bool = False):
        env = gym.make(env_name, render_mode="rgb_array" if render else None, continuous=True)

        env = RecordEpisodeStatistics(env)

        # convert rgb to grayscale
        env = GrayScaleObservation(env)

        env = FrameStack(env, num_stack=4)
        return env

    log_string = "{}_{}_{}_s{}_l{}_alr{}_clr{}_b{}_d{}".format(
        exp_name or "offpolicy_ac",
        env_name,
        actor_gradient_type,
        hidden_size,
        num_layers,
        actor_learning_rate,
        critic_learning_rate,
        batch_size,
        discount,
    )

    if use_entropy_bonus:
        log_string += f"_t{temperature}"

    if use_soft_target_update:
        log_string += f"_stu{soft_target_update_rate}"
    else:
        log_string += f"_htu{target_update_period}"

    if target_critic_backup_type != "mean":
        log_string += f"_{target_critic_backup_type}"

    return {
        "agent_kwargs": {
            "make_critic": make_critic,
            "make_critic_optimizer": make_actor_optimizer,
            "make_critic_schedule": make_lr_schedule,
            "make_actor": make_actor,
            "make_actor_optimizer": make_critic_optimizer,
            "make_actor_schedule": make_lr_schedule,
            "num_critic_updates": num_critic_updates,
            "discount": discount,
            "actor_gradient_type": actor_gradient_type,
            "num_actor_samples": num_actor_samples,
            "num_critic_updates": num_critic_updates,
            "num_critic_networks": num_critic_networks,
            "target_critic_backup_type": target_critic_backup_type,
            "use_entropy_bonus": use_entropy_bonus,
            "backup_entropy": backup_entropy,
            "temperature": temperature,
            "target_update_period": target_update_period
            if not use_soft_target_update
            else None,
            "soft_target_update_rate": soft_target_update_rate
            if use_soft_target_update
            else None,
        },
        "replay_buffer_capacity": replay_buffer_capacity,
        "log_name": log_string,
        "total_steps": total_steps,
        "random_steps": random_steps,
        "training_starts": training_starts,
        "ep_len": ep_len,
        "batch_size": batch_size,
        "make_env": make_env,
    }
