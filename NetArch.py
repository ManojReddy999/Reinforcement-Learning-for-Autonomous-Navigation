import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthMap3DFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, temporal_depth=4, feature_dim=256):
        super(DepthMap3DFeatureExtractor, self).__init__()
        
        self.temporal_depth = temporal_depth  # Number of frames

        # 3D Convolutional Layers
        self.cnn = nn.Sequential(
            nn.Conv3d(in_channels=input_channels, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 2, 2)),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(256),
            nn.ReLU(),

            # nn.Conv3d(in_channels=256, out_channels=512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            # nn.BatchNorm3d(512),
            # nn.ReLU(),
            
            nn.Flatten()
        )
        
        # Compute the number of features after convolutional layers
        with th.no_grad():
            # Create a sample input tensor with the expected shape
            # Shape: (1, 1, temporal_depth, 128, 128)
            sample_input = th.zeros(1, input_channels, temporal_depth, 128, 128)
            n_flatten = self.cnn(sample_input).shape[1]
        
        # Fully Connected Layers
        self.fc = nn.Sequential(
            nn.Linear(n_flatten, feature_dim),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )
    
    def forward(self, x):
        # x shape: (batch_size, 1, temporal_depth, 128, 128)
        features = self.cnn(x)  # (batch_size, n_flatten)
        features = self.fc(features)  # (batch_size, feature_dim)
        return features

# Wrapper for SB3
class DepthMap3DFeatureExtractorWrapper(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, temporal_depth=4):
        """
        Wrapper for the 3D CNN feature extractor.

        Args:
            observation_space (gym.Space): The observation space of the environment.
            features_dim (int): The dimension of the output feature vector.
            temporal_depth (int): Number of temporal frames in the observation.
        """
        # Ensure the observation space is a Dict with 'img' key
        assert isinstance(observation_space, spaces.Dict), "Observation space must be a Dict space"
        assert 'img' in observation_space.spaces, "Observation space must have 'img' key"
        
        super(DepthMap3DFeatureExtractorWrapper, self).__init__(observation_space, features_dim)
        
        self.feature_extractor = DepthMap3DFeatureExtractor(
            input_channels=1,
            temporal_depth=temporal_depth,
            feature_dim=features_dim
        )
    
    def forward(self, observations):
        """
        Forward pass for the feature extractor.

        Args:
            observations (dict): A dictionary containing the 'img' key with depth images.

        Returns:
            torch.Tensor: Extracted feature vector.
        """
        depth_maps = observations['img']  # Shape: (batch_size, temporal_depth, 128, 128, 1)
        depth_maps = self.normalize_depth_maps(depth_maps)  # Normalize
        depth_maps = depth_maps.permute(0, 4, 1, 2, 3)  # (batch_size, 1, temporal_depth, 128, 128)
        features = self.feature_extractor(depth_maps)  # (batch_size, features_dim)
        return features
    
    def normalize_depth_maps(self, depth_maps):
        """
        Normalize depth maps to range [0, 1].

        Args:
            depth_maps (torch.Tensor): Tensor containing depth maps.

        Returns:
            torch.Tensor: Normalized depth maps.
        """
        min_val = depth_maps.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0].min(dim=-3, keepdim=True)[0]
        max_val = depth_maps.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0].max(dim=-3, keepdim=True)[0]
        return (depth_maps - min_val) / (max_val - min_val + 1e-6)  # Avoid division by zero






class Custom3DConvFeatureExtractor(nn.Module):
    def __init__(self, input_channels=1, temporal_depth=6, features_dim=256):
        super(Custom3DConvFeatureExtractor, self).__init__()
        
        self.temporal_depth = temporal_depth  # Number of frames

        # 3D Convolutional Layers
        self.conv3d_1 = nn.Conv3d(in_channels=input_channels,
                                  out_channels=32,
                                  kernel_size=(3, 5, 5),
                                  stride=(1, 2, 2),
                                  padding=(1, 2, 2))  # Output: (32, 6, 64, 64)
        self.bn1 = nn.BatchNorm3d(32)
        
        self.conv3d_2 = nn.Conv3d(in_channels=32,
                                  out_channels=64,
                                  kernel_size=(3, 3, 3),
                                  stride=(1, 2, 2),
                                  padding=(1, 1, 1))  # Output: (64, 6, 32, 32)
        self.bn2 = nn.BatchNorm3d(64)
        
        self.conv3d_3 = nn.Conv3d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=(3, 3, 3),
                                  stride=(2, 2, 2),
                                  padding=(1, 1, 1))  # Output: (128, 3, 16, 16)
        self.bn3 = nn.BatchNorm3d(128)
        
        self.conv3d_4 = nn.Conv3d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=(3, 3, 3),
                                  stride=(2, 2, 2),
                                  padding=(1, 1, 1))  # Output: (256, 2, 8, 8)
        self.bn4 = nn.BatchNorm3d(256)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 2 * 8 * 8, features_dim)
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        """
        x: Tensor of shape (batch_size, temporal_depth, height, width, channels)
        Expected input shape: (batch_size, 6, 128, 128, 1)
        """
        # Rearrange to (batch_size, channels, temporal_depth, height, width)
        x = x.permute(0, 4, 1, 2, 3)  # (batch_size, 1, 6, 128, 128)
        
        x = F.relu(self.bn1(self.conv3d_1(x)))  # (batch_size, 32, 6, 64, 64)
        x = F.relu(self.bn2(self.conv3d_2(x)))  # (batch_size, 64, 6, 32, 32)
        x = F.relu(self.bn3(self.conv3d_3(x)))  # (batch_size, 128, 3, 16, 16)
        x = F.relu(self.bn4(self.conv3d_4(x)))  # (batch_size, 256, 2, 8, 8)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch_size, 256*2*8*8)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))    # (batch_size, feature_dim)
        x = self.dropout(x)
        
        return x  # Feature vector of size (batch_size, feature_dim)


class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim: int = 256):
        # Assuming observation_space.shape is (6, 128, 128, 1)
        super(CustomFeaturesExtractor, self).__init__(observation_space, features_dim)
        # Initialize the 3D Conv Feature Extractor
        self.feature_extractor = Custom3DConvFeatureExtractor(
            input_channels=1,
            temporal_depth=6,
            features_dim=features_dim
        )
        
    def forward(self, observations):
        # observations shape: (batch_size, 6, 128, 128, 1)
        features = self.feature_extractor(observations['img'])
        return features

# Define a custom policy with the feature extractor
# from stable_baselines3.common.policies import ActorCriticPolicy

# class CustomSACPolicy(ActorCriticPolicy):
#     def __init__(self, *args, **kwargs):
#         super(CustomSACPolicy, self).__init__(
#             *args,
#             **kwargs,
#             features_extractor_class=CustomFeaturesExtractor,
#             features_extractor_kwargs=dict(feature_dim=256),
#         )
#         # Adjust the policy network to accept the new feature dimension
#         self.actor = nn.Sequential(
#             self.features_extractor,
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, self.action_space.shape[0]),
#         )
#         self.critic = nn.Sequential(
#             self.features_extractor,
#             nn.Linear(256, 128),
#             nn.ReLU(),
#             nn.Linear(128, 1),
#         )




import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import MultiInputPolicy
from gymnasium import spaces

class CustomMultiInputExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multi-input environments.
    Processes vector and image observations separately and combines them.
    """
    def __init__(self, observation_space, features_dim=128):
        # Call the BaseFeaturesExtractor's constructor
        super(CustomMultiInputExtractor, self).__init__(observation_space, features_dim)
        
        # Define the CNN for image input
        n_input_channels = observation_space['img'].shape[-1]  # number of channels in depth image
        self.cnn = nn.Sequential(
            # First block of Convolution
            nn.Conv2d(n_input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Second block of Convolution
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), 
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Third block of Convolution
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # nn.AdaptiveAvgPool2d(1),  # Global average pooling layer
            nn.Flatten()
        )
        
        # Compute the size of the CNN output
        with th.no_grad():
            # Process sample depth input to determine output size
            sample_depth = observation_space['img'].sample()[None, :, :, 0]
            sample_depth = th.tensor(sample_depth).float()
            sample_depth = self.normalize_depth_map(sample_depth)
            sample_depth = sample_depth.unsqueeze(1)
            cnn_output_size = self.cnn(sample_depth).shape[1]
        
        # Define the state processing network (fully connected layers)
        n_state_inputs = observation_space['vec'].shape[0]
        self.state_net = nn.Sequential(
            nn.Linear(n_state_inputs, 64),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.ReLU(),
            # nn.Dropout(0.3)
        )
        
        # Combine the CNN and state outputs
        self.combined_fc = nn.Sequential(
            nn.Linear(cnn_output_size + 64, features_dim),
            nn.ReLU()
            # nn.Linear(features_dim, features_dim),
            # nn.ReLU()
        )
    
    def forward(self, observations):
        # Process the image observations, make sure to remove the last dimension if it's there

        depth_map = observations['img'][:, :, :, 0]
        depth_map = self.normalize_depth_map(depth_map)
        depth_map = depth_map.unsqueeze(1)
        image_features = self.cnn(depth_map)
        
        # Process the vector observations
        vector_features = self.state_net(observations['vec'])
        # vector_features = observations['vec']
        
        # Concatenate both feature outputs
        combined_features = th.cat((image_features, vector_features), dim=1)
        
        # Pass through the final fully connected layer
        return self.combined_fc(combined_features)
    
    def normalize_depth_map(self, depth_map):
        min_val = depth_map.min()
        max_val = depth_map.max()
        return (depth_map - min_val) / (max_val - min_val)


import torch as th
import torch.nn as nn
from gymnasium.spaces import Box
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class DepthMapFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: Box, features_dim: int = 128):
        """
        Initializes the feature extractor with CNN and LSTM layers.

        Args:
            observation_space (Box): The observation space containing the depth images.
            features_dim (int): The dimension of the extracted features.
        """
        # Ensure the observation space has the expected shape
        # assert isinstance(observation_space, dict), "Observation space should be a dict."
        # assert 'img' in observation_space, "Observation space must contain 'img' key."
        
        # Initialize the BaseFeaturesExtractor
        super(DepthMapFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Define CNN to extract spatial features from each depth image
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),  # (1, 128, 128) -> (32, 64, 64)
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # (32, 64, 64) -> (64, 32, 32)
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),  # (64, 32, 32) -> (128, 16, 16)
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1),  # (128,16,16) -> (256,8,8)
            nn.ReLU(),
            nn.Flatten()  # (256,8,8) -> 256*8*8=16384
        )

        # Fully connected layer to reduce CNN output to a manageable size
        self.cnn_fc = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU()
        )
        
        # Determine the size of the CNN output to configure the LSTM
        with th.no_grad():
            # Sample a single observation to infer the CNN output size
            sample_depth = observation_space['img'].sample()[None, :, :, :, :]  # Shape: (1, 3, 128, 128, 1)
            sample_depth = th.tensor(sample_depth).float()  # Convert to float tensor
            sample_depth = sample_depth.permute(0, 1, 4, 2, 3)  # Rearrange to (1, 3, 1, 128, 128)
            cnn_output = self.cnn(sample_depth.view(-1, 1, 128, 128))  # Pass through CNN: (3, 16384)
            cnn_output = self.cnn_fc(cnn_output)
            cnn_feature_dim = cnn_output.shape[1]  # 512
        
        # Define LSTM to process the sequence of CNN features
        self.lstm_hidden_size = 128  # You can adjust this
        self.lstm = nn.LSTM(input_size=cnn_feature_dim,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=3,
                            batch_first=True)
        
        # Define a linear layer to project LSTM output to features_dim
        self.linear = nn.Sequential(
            nn.Linear(self.lstm_hidden_size, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations):
        """
        Forward pass through the feature extractor.

        Args:
            observations (dict): A dictionary containing the 'img' key with depth images.

        Returns:
            th.Tensor: Extracted features of shape (batch_size, features_dim).
        """
        # Extract the image sequence
        img_seq = observations['img']  # Expected shape: (batch_size, seq_len=3, channels=1, height=128, width=128)
        
        # Rearrange dimensions to (batch_size, 3, 1, 128, 128)
        img_seq = img_seq.permute(0, 1, 4, 2, 3)  # (batch_size, 3, 1, 128, 128)

        batch_size, seq_len, channels, height, width = img_seq.shape
        
        # Normalize the depth maps
        img_seq = self.normalize_depth_map(img_seq)
        
        # Reshape to (batch_size * seq_len, channels, height, width) for CNN processing
        img_seq = img_seq.view(batch_size * seq_len, channels, height, width)
        
        # Pass through CNN to extract spatial features
        cnn_features = self.cnn(img_seq)  # Shape: (batch_size * seq_len, cnn_feature_dim)
        
        # Reshape to (batch_size, seq_len, cnn_feature_dim) for LSTM processing
        cnn_features = cnn_features.view(batch_size, seq_len, -1)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_features)  # lstm_out shape: (batch_size, seq_len, lstm_hidden_size)
        
        # Use the last time step's output as the sequence representation
        lstm_features = lstm_out[:, -1, :]  # Shape: (batch_size, lstm_hidden_size)
        
        # Pass through the linear layer to get the final features
        features = self.linear(lstm_features)  # Shape: (batch_size, features_dim)
        
        return features
    
    @staticmethod
    def normalize_depth_map(depth_map):
        """
        Normalizes the depth maps.

        Args:
            depth_map (th.Tensor): Tensor of depth maps.

        Returns:
            th.Tensor: Normalized depth maps.
        """
        # Compute mean and std across batch, sequence, height, and width
        mean = depth_map.mean(dim=[1, 2, 3, 4], keepdim=True)
        std = depth_map.std(dim=[1, 2, 3, 4], keepdim=True) + 1e-6
        return (depth_map - mean) / std
