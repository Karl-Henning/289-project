from typing import Optional, Tuple, List

from torch import nn

import torch
from torch import distributions

from infrastructure import pytorch_util as ptu
from infrastructure.distributions import make_tanh_transformed, make_multi_normal

from env_configs.dqn_car_racing_config import PreprocessCarRacing

class CNNPolicy(nn.Module):
    """
    Base MLP policy, which can take an observation and output a distribution over actions.

    This class implements `forward()` which takes a (batched) observation and returns a distribution over actions.
    """

    def __init__(
        self,
        ac_dim: int,
        obs_shape: Tuple[int, int, int], # Shape of the input image (height, width, channels)
        discrete: bool,

        n_conv_layers: int,
        conv_channels: List[int],
        conv_kernel_sizes: List[int],
        conv_strides: List[int],

        n_fc_layers: int,
        fc_layer_size: int,
        use_tanh: bool = False,
        state_dependent_std: bool = False,
        fixed_std: Optional[float] = None,
    ):
        super().__init__()

        self.use_tanh = use_tanh
        self.discrete = discrete
        self.state_dependent_std = state_dependent_std
        self.fixed_std = fixed_std

        # pre-process the input image
        self.preprocess = PreprocessCarRacing()

        # Define convolutional layers
        conv_layers = []
        in_channels = obs_shape[0]  # Assuming input is grayscale, hence 1 channel
        for out_channels, kernel_size, stride in zip(conv_channels, conv_kernel_sizes, conv_strides):
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride))
            conv_layers.append(nn.ReLU())
            in_channels = out_channels
        self.conv_layers = nn.Sequential(*conv_layers).to(ptu.device)

        # Calculate the shape of the output from convolutional layers
        conv_output_dim = 4096

        # Define fully connected layers
        if discrete:
            self.fc_layers = ptu.build_mlp(
                input_size=conv_output_dim,
                output_size=ac_dim,
                n_layers=n_fc_layers,
                size=fc_layer_size,
            ).to(ptu.device)
        else:
            if self.state_dependent_std:
                assert fixed_std is None
                self.fc_layers = ptu.build_mlp(
                    input_size=conv_output_dim,
                    output_size=2*ac_dim,
                    n_layers=n_fc_layers,
                    size=fc_layer_size,
                ).to(ptu.device)
            else:
                self.fc_layers = ptu.build_mlp(
                    input_size=conv_output_dim,
                    output_size=ac_dim,
                    n_layers=n_fc_layers,
                    size=fc_layer_size,
                ).to(ptu.device)

                if self.fixed_std:
                    self.std = 0.1
                else:
                    self.std = nn.Parameter(
                        torch.full((ac_dim,), 0.0, dtype=torch.float32, device=ptu.device)
                    )


    def forward(self, obs: torch.FloatTensor) -> distributions.Distribution:
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        # obs = obs.permute(0, 3, 1, 2)  # Assuming obs is in NHWC format, convert to NCHW
        preprocessed_obs = self.preprocess(obs)
        conv_output = self.conv_layers(preprocessed_obs)
        conv_output_flat = conv_output.view(conv_output.size(0), -1)  # Flatten convolutional output

        if self.discrete:
            logits = self.fc_layers(conv_output_flat)
            action_distribution = distributions.Categorical(logits=logits)
        else:
            if self.state_dependent_std:
                mean, std = torch.chunk(self.fc_layers(conv_output_flat), 2, dim=-1)
                std = torch.nn.functional.softplus(std) + 1e-2
            else:
                mean = self.fc_layers(conv_output_flat)
                if self.fixed_std:
                    std = self.std
                else:
                    std = torch.nn.functional.softplus(self.std) + 1e-2

            if self.use_tanh:
                action_distribution = make_tanh_transformed(mean, std)
            else:
                action_distribution = make_multi_normal(mean, std)

        return action_distribution
 
