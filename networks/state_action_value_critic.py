import torch
from torch import nn

import infrastructure.pytorch_util as ptu
from env_configs.dqn_car_racing_config import PreprocessCarRacing

class StateActionCritic(nn.Module):
    def __init__(self, ob_shape, ac_dim, n_layers, size):
        super().__init__()

        assert ob_shape == (
            4,
            96,
            96,
        ), f"Observation shape: {ob_shape}"

        # Define the CNN layers
        self.conv_layers = nn.Sequential(
            PreprocessCarRacing(),
            nn.Conv2d(in_channels=4, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
        ).to(ptu.device)

        conv_output_size = 3136
        
        # # Calculate the size of the flattened output of the convolutional layers
        # with torch.no_grad():
        #     dummy_input = torch.zeros(1, *ob_shape)
        #     conv_out_size = self._get_conv_out(dummy_input)

        self.flatten = nn.Flatten()

        # Define the fully connected layers
        self.fc_layers = ptu.build_mlp(
            input_size=conv_output_size + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)
    
    def forward(self, obs, acs):
        # print(obs.shape)
        # print(acs.shape)
        if len(obs.shape) == 5 and len(acs.shape) == 3:
            cnn_results = [None] * obs.shape[0]
            for i in range(obs.shape[0]):
                cnn_out = self.conv_layers(obs[i])
                cnn_out = self.flatten(cnn_out)
                cnn_results[i] = cnn_out
            cnn_results = torch.stack(cnn_results)
        else:
            cnn_results = self.conv_layers(obs)
            cnn_results = self.flatten(cnn_results)

        combined_input = torch.cat([cnn_results, acs], dim=-1)
        q_values = self.fc_layers(combined_input).squeeze(-1)

        return q_values
