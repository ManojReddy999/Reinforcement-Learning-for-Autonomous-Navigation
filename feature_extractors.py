from torch_geometric.data import Data, Batch
import numpy as np
from scipy.spatial import KDTree
import torch
import torch.nn as nn
import torch_geometric
from torch_geometric.nn import EdgeConv, SAGEConv, global_mean_pool, radius_graph, knn_graph
from torch.nn.functional import normalize, relu
from torch_scatter import scatter_max, scatter_mean, scatter_add, scatter_min
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import open3d as o3d

def point_cloud_to_graph(point_clouds, method="knn", k=32, radius=1.5, z_min=0, z_max=20.0, normalize=False):
    """
    Converts a batch of point clouds (batch, N, 3) into a graph representation.
    
    Parameters:
    - point_clouds: Tensor of shape (batch, N, 3), where batch is the number of point clouds.
    - method: "knn" or "radius" for graph construction.
    - k: Number of neighbors for KNN graph or maximum number of neighbours for radius graph.
    - radius: Radius for radius graph.
    
    Returns:
    - edge_index: Connectivity information for graph (2, num_edges).
    - point_clouds: Batched point clouds.
    - batch: Tensor indicating which points belong to which batch element.
    """

    batch_size, num_points, dim = point_clouds.shape  # (B, N, 3)

    # Reshape into (B*N, 3) to handle it like a flat list
    point_clouds = point_clouds.view(-1, dim)

    # Create batch tensor (Assign each point to its corresponding batch)
    batch = torch.arange(batch_size, device=point_clouds.device).repeat_interleave(num_points)

    # Graph Construction
    if method == "knn":
        edge_index = knn_graph(point_clouds, k=k, batch=batch)
    elif method == "radius":
        edge_index = radius_graph(point_clouds, r=radius, batch=batch, max_num_neighbors=k)
    else:
        raise ValueError("Invalid method. Choose 'knn' or 'radius'.")

    # Normalize the filtered point cloud batch-wise (Min-Max Normalization)
    if normalize:
        min_xyz = scatter_min(point_clouds, batch, dim=0)[0]
        max_xyz = scatter_max(point_clouds, batch, dim=0)[0]
        
        point_clouds = (point_clouds - min_xyz[batch]) / (max_xyz[batch] - min_xyz[batch] + 1e-6)

    return edge_index, point_clouds, batch


class PointGNN(nn.Module):
    """Graph Convolution layer that updates vertex features using edge features"""
    def __init__(self, in_features, out_features):
        super(PointGNN, self).__init__()
        self.edge_fc = nn.Linear(in_features,out_features)

        # self.edge_fc = nn.Sequential(nn.Linear(2*in_features, out_features),
        #                              nn.LayerNorm(out_features),
        #                              nn.ReLU(),
        #                              nn.Linear(out_features,out_features),
        #                             #  nn.BatchNorm1d(out_features),
        #                             #  nn.ReLU()
        #                             )

        self.node_fc = nn.Sequential(nn.Linear(in_features+out_features, out_features),
                                     nn.LayerNorm(out_features),
                                     nn.ReLU(),
                                     nn.Linear(out_features,out_features),
                                    #  nn.BatchNorm1d(out_features),
                                    #  nn.ReLU()
                                    )

        # self.node_fc = nn.Linear(in_features+out_features, out_features)

        # Learnable scaling parameter
        # self.alpha = nn.Parameter(torch.full((out_features,), 10.0))  # Start with scale=1

    def forward(self, vertex_features, edge_index):
        src, dst = edge_index
        
        # Compute the edge features and aggregate them
        edge_features = vertex_features[src] - vertex_features[dst]
        # edge_features = torch.cat([vertex_features[src], vertex_features[dst]], dim=-1)
        edge_features = self.edge_fc(edge_features)

        # with torch.amp.autocast('cuda'):
        aggregated_edge_features = scatter_max(edge_features, dst, dim=0)[0]

        # Combine the vertex features, aggregated edge features and process them
        combined_features = torch.cat([vertex_features, aggregated_edge_features], dim=-1)
        updated_vertex_features = self.node_fc(combined_features)
        # updated_vertex_features = aggregated_edge_features

                # Apply Scaling Factor
        # updated_vertex_features = self.alpha * updated_vertex_features  # <--- SCALE OUTPUT

        return updated_vertex_features

class EdgeConvWrapper(nn.Module):
    """Edge Convolution Layer using PyG's built-in EdgeConv."""
    def __init__(self, in_features, out_features):
        super(EdgeConvWrapper, self).__init__()
        self.edge_conv = EdgeConv(nn.Sequential(
            nn.Linear(2 * in_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU()
        ))

    def forward(self, vertex_features, edge_index):
        return self.edge_conv(vertex_features, edge_index)


class PointGNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,num_layers=3,use_edgeconv=False):
        super(PointGNNFeatureExtractor, self).__init__()

        self.use_edgeconv = use_edgeconv
        self.layers = nn.ModuleList()

        if use_edgeconv:
            self.layers.append(EdgeConvWrapper(input_dim, hidden_dim))
            for _ in range(num_layers - 2):
                self.layers.append(EdgeConvWrapper(hidden_dim, 2*hidden_dim))
                hidden_dim = 2*hidden_dim
            self.layers.append(EdgeConvWrapper(hidden_dim, output_dim))

        else:
            self.layers.append(PointGNN(input_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.layers.append(PointGNN(hidden_dim,2*hidden_dim))
                hidden_dim = 2*hidden_dim
            self.layers.append(PointGNN(hidden_dim,output_dim))

    def forward(self, edge_index, vertex_features, batch):

        # vertex_features, edge_index, batch = data.x, data.edge_index, data.batch
        for layer in self.layers:
            vertex_features = layer(vertex_features, edge_index)

        #Aggregate features per batch
        batch_features = scatter_max(vertex_features, batch, dim=0)[0]

        return batch_features
    
class PointGNNFeatureExtractorWrapper(BaseFeaturesExtractor):
    def __init__(self, observation_space, input_dim = 3, hidden_dim = 64, output_dim = 128, features_dim=256, num_layers=3,use_edgeconv=False):
        super(PointGNNFeatureExtractorWrapper, self).__init__(observation_space, features_dim=features_dim)

        self.feature_extractor = PointGNNFeatureExtractor(input_dim, hidden_dim, output_dim, num_layers,use_edgeconv)

        # Define the state processing network (fully connected layers)
        if 'vec' in observation_space:
            n_state_inputs = observation_space['vec'].shape[0]
            self.state_net = nn.Sequential(
                nn.Linear(n_state_inputs, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
            )

            # Combine the GNN and state outputs
            self.combined_fc = nn.Sequential(
                nn.Linear(output_dim + 64, features_dim),
                nn.ReLU(),
                nn.Linear(features_dim, features_dim),
                nn.ReLU()
            )

    def forward(self, observations):
        # convert point cloud to graph

        edge_index, vertex_features, batch = point_cloud_to_graph(observations['cam'], method='knn', k=4, normalize=False)
        cam_features = self.feature_extractor(edge_index, vertex_features, batch)

        # Process the vector observations
        if 'vec' in observations:
            vector_features = self.state_net(observations['vec'])
            combined_features = torch.cat((cam_features, vector_features), dim=1)
            return self.combined_fc(combined_features)
        
        return cam_features
    
def count_neighbors(edge_index, num_nodes):
    """Counts the number of neighbors for each node."""
    neighbor_counts = torch.bincount(edge_index[1], minlength=num_nodes)
    return neighbor_counts


def visualize_graph(point_cloud, edges):
    """
    Visualizes the point cloud with edges using Open3D.
    Fixes the issue of Tensor type mismatch.
    """
    # Convert to NumPy if input is a PyTorch tensor
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.cpu().numpy()
    
    if isinstance(edges, torch.Tensor):
        edges = edges.cpu().numpy().T  # Convert PyTorch Tensor to NumPy array and transpose

    # Convert point cloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Create line set for edges
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(point_cloud)
    line_set.lines = o3d.utility.Vector2iVector(edges.astype(np.int32))  # Ensure integer type

    # Add color to edges (optional)
    colors = [[1, 0, 0] for _ in range(len(edges))]  # Red edges
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # Visualize
    o3d.visualization.draw_geometries([pcd, line_set])


class CNNFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for multi-input environments.
    Processes vector and image observations separately and combines them.
    """
    def __init__(self, observation_space, features_dim=128):
        # Call the BaseFeaturesExtractor's constructor
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)
        
        # Define the CNN for image input
        n_input_channels = observation_space['cam'].shape[-1]  # number of channels in depth image
        self.cnn = nn.Sequential(
            # First block of Convolution
            nn.Conv2d(n_input_channels, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2),

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

        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.min_pool = lambda x: -torch.nn.functional.adaptive_max_pool2d(-x, 1)  # Global min pooling layer
        # pooled_channels = 128*2
        
        # Compute the size of the CNN output
        with torch.no_grad():
            # Process sample depth input to determine output size
            sample_depth = observation_space['cam'].sample()[None, :, :, 0]
            sample_depth = torch.tensor(sample_depth).float()
            sample_depth = self.normalize_depth_map(sample_depth)
            sample_depth = sample_depth.unsqueeze(1)
            cnn_output_size = self.cnn(sample_depth).shape[1]
        
        vec_feature_size = 0
        # Define the state processing network (fully connected layers)
        if 'vec' in observation_space.keys():
        # print("creating state_net")
            n_state_inputs = observation_space['vec'].shape[0]
            self.state_net = nn.Sequential(
                nn.Linear(n_state_inputs, 64),
                nn.ReLU(),
                # nn.Dropout(0.3),
                nn.Linear(64, 64),
                nn.ReLU(),
                # nn.Dropout(0.3)
            )
            vec_feature_size += 64
        
        # Combine the CNN and state outputs
        combined_input = cnn_output_size + vec_feature_size
        # combined_input = pooled_channels + vec_feature_size
        self.combined_fc = nn.Sequential(
            nn.Linear(combined_input, features_dim),
            nn.ReLU()
            # nn.Linear(features_dim, features_dim),
            # nn.ReLU()
        )
    
    def forward(self, observations):
        # Process the image observations, make sure to remove the last dimension if it's there

        depth_map = observations['cam'][:, :, :, 0]
        depth_map = self.normalize_depth_map(depth_map)
        depth_map = depth_map.unsqueeze(1)
        features = self.cnn(depth_map)

        # global avg & min pool
        # avg = self.avg_pool(features).view(features.size(0), -1)
        # min_ = self.min_pool(features).view(features.size(0), -1)
        # features = torch.cat([avg, min_], dim=1)
        
        # Process the vector observations
        if "vec" in observations:
            vector_features = self.state_net(observations['vec'])
        # Concatenate both feature outputs
            features = torch.cat((features, vector_features), dim=1)
   
        # Pass through the final fully connected layer
        return self.combined_fc(features)
    
    def normalize_depth_map(self, depth_map):
        min_val = depth_map.min()
        max_val = depth_map.max()
        return (depth_map - min_val) / (max_val - min_val)



# --------------------------------------------------------------------------
# 1) CNN for multi-frame depth
#    This is a Nature-like CNN adapted for 5 frames of grayscale depth input.
# --------------------------------------------------------------------------
class DepthHistoryCNN(nn.Module):
    """
    Handles both cases transparently:

      • 1-frame depth  -> input shape  [B, 1, H, W]
      • 5-frame depth  -> input shape  [B, 5, H, W]

    The number of channels is inferred at run time.
    """
    def __init__(self, in_channels: int, base_channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            # Conv1
            nn.Conv2d(in_channels, base_channels, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            # Conv2
            nn.Conv2d(base_channels, base_channels*2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # Conv3
            nn.Conv2d(base_channels*2, base_channels*2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.output_channels = base_channels*2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, C, H, W] where C == 1 or 5
        return self.net(x)


# --------------------------------------------------------------------------
# 2) MLP for velocity history
# --------------------------------------------------------------------------
class VelocityHistoryMLP(nn.Module):
    """
    If you have velocity history of shape (10,) => 5 timesteps × 2 dims each.
    Produces an embedding of dimension 'token_dim'.
    """
    def __init__(self, input_dim, token_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, token_dim),
            nn.ReLU()
        )
        self.output_dim = token_dim

    def forward(self, vel_hist):
        # shape: [B, input_dim]
        return self.mlp(vel_hist)  # [B, token_dim]

# --------------------------------------------------------------------------
# 3) The cross-modal Transformer
#    We'll replicate the idea of LocoTransformer: 
#     - We produce a single "token" from velocity MLP
#     - We produce multiple tokens from CNN features 
#     - Then we run them through a TransformerEncoder
# --------------------------------------------------------------------------
class LocoTransformerCore(nn.Module):
    """
    A small wrapper that builds a stack of TransformerEncoderLayers
    to handle (num_tokens, batch, token_dim).
    """
    def __init__(self, token_dim=128, n_heads=4, n_layers=2, feedforward_dim=256):
        super().__init__()
        # Build a standard PyTorch TransformerEncoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation='relu',
            batch_first=False  # we'll pass (seq, batch, dim)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.token_dim = token_dim

    def forward(self, tokens):
        """
        tokens shape: [num_tokens, batch_size, token_dim]
        returns same shape
        """
        return self.transformer(tokens)

# --------------------------------------------------------------------------
# 4) The final Feature Extractor that you register in SB3
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# 3) Feature extractor that now auto-detects 1-frame vs 5-frame depth
# --------------------------------------------------------------------------
class HistoryLocoTransformerExtractor(BaseFeaturesExtractor):
    """
    Works with either:

      • depth history  (5, 64, 128, 1)   → 5-frame mode
      • single depth   (64, 128, 1)      → 1-frame mode

    The rest of the API is unchanged.
    """
    def __init__(
        self,
        observation_space,
        features_dim: int = 128,
        token_dim: int = 64,
        n_heads: int = 1,
        n_layers: int = 2,
        feedforward_dim: int = 256,
        vel_hist_dim: int = 10,
        base_channels: int = 32,
    ):
        super().__init__(observation_space, features_dim)

        # ---------- Which modalities are present? ----------
        self.use_velocity = "vec" in observation_space.spaces
        self.use_cam      = "cam" in observation_space.spaces
        # assert self.use_cam, "`cam` entry not found in observation_space"

        cam_shape = observation_space["cam"].shape   # (… , 1)
        #   • single-frame: (H, W, 1)           len == 3
        #   • 5-history  : (5, H, W, 1)        len == 4
        self.n_frames = cam_shape[0] if len(cam_shape) == 4 else 1

        # ---------- Visual pathway ----------
        self.depth_cnn = DepthHistoryCNN(in_channels=self.n_frames, base_channels=base_channels)
        self.up_conv = nn.Conv2d(
            in_channels=self.depth_cnn.output_channels,
            out_channels=token_dim,
            kernel_size=1,
        )

        # ---------- Velocity pathway ----------
        if self.use_velocity:
            vel_dim = observation_space["vec"].shape[0] 
            self.vel_mlp = VelocityHistoryMLP(
                input_dim=vel_dim,
                token_dim=token_dim,
            )

        # ---------- Cross-modal transformer ----------
        self.xform = LocoTransformerCore(
            token_dim=token_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            feedforward_dim=feedforward_dim,
        )

        # ---------- Final head ----------
        pooled_dim = token_dim * (2 if self.use_velocity else 1)
        self.final_head = nn.Sequential(
            nn.Linear(pooled_dim, features_dim),
            nn.ReLU(),
        )

    # ------------------------------------------------------------------
    # forward()
    # ------------------------------------------------------------------
    def forward(self, observations):
        cam_raw = observations["cam"]                      # [B, …, 1]

        # Re-arrange to [B, C, H, W]  where C = 1 | 5
        if self.n_frames == 1:
            # cam_raw = [B, H, W, 1]  → [B, 1, H, W]
            cam_input = cam_raw.squeeze(-1).unsqueeze(1)
        else:
            # cam_raw = [B, 5, H, W, 1] → [B, 5, H, W]
            cam_input = cam_raw.squeeze(-1)

        cam_input = self._normalize_depth(cam_input)
        feats     = self.depth_cnn(cam_input)              # [B, C’, h’, w’]
        tokens_2d = self.up_conv(feats)                    # [B, T, h’, w’]

        # Flatten spatial dims -> [Npatch, B, token_dim]
        B, T, h, w = tokens_2d.shape
        vis_tokens = tokens_2d.view(B, T, h * w).permute(2, 0, 1).contiguous()

        # Velocity token (optional) -> [1, B, token_dim]
        if self.use_velocity:
            vel_token = self.vel_mlp(observations["vec"].float()).unsqueeze(0)
            all_tokens = torch.cat([vel_token, vis_tokens], dim=0)
        else:
            all_tokens = vis_tokens

        # Cross-modal transformer
        out_tokens = self.xform(all_tokens)                # same shape

        # Pool
        if self.use_velocity:
            pooled = torch.cat(
                [out_tokens[0],                    # velocity
                 out_tokens[1:].mean(dim=0)],      # mean of visuals
                dim=-1,
            )
        else:
            pooled = out_tokens.mean(dim=0)               # [B, token_dim]

        return self.final_head(pooled)                    # [B, features_dim]

    # ------------------------------------------------------------------
    # utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_depth(depth_map: torch.Tensor) -> torch.Tensor:
        """
        Min–max normalise **per frame** (handles 1- or 5-frame inputs).
        depth_map: [B, C, H, W]
        """
        min_val = depth_map.amin(dim=(2, 3), keepdim=True)
        max_val = depth_map.amax(dim=(2, 3), keepdim=True)
        return (depth_map - min_val) / (max_val - min_val + 1e-6)

