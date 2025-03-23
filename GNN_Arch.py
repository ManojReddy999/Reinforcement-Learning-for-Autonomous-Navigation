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


# def point_cloud_to_graph(point_clouds, radius=0.015, max_num_neighbors=8):
#     # Flatten batch of point clouds and add batch indices
#     batch_size, num_points, dim = point_clouds.size()

#     # Filter out points based only on z-values
#     valid_mask = point_clouds[..., 2] < point_clouds[..., 2].max()  # Only filter based on z
#     point_clouds = point_clouds[valid_mask]

#     # Create batch indices
#     batch_indices = torch.arange(batch_size, device=point_clouds.device).repeat_interleave(num_points)[valid_mask.view(-1)]
#     # batch_indices = torch.arange(batch_size, device=point_clouds.device).repeat_interleave(num_points)
#     point_clouds_flat = point_clouds.view(-1, dim)

#     # Construct edge indices for the entire batch
#     # edge_index = radius_graph(point_clouds_flat, r=radius, batch=batch_indices, loop=False, max_num_neighbors=max_num_neighbors)

#         # Use k-NN graph instead of radius graph
#     edge_index = torch_geometric.nn.knn_graph(point_clouds_flat, k=max_num_neighbors, batch=batch_indices, loop=False)

#         # Normalize point cloud
#     # mean = point_clouds_flat.mean(dim=0, keepdim=True)
#     # std = point_clouds_flat.std(dim=0, keepdim=True) + 1e-6
#     # point_clouds_flat = (point_clouds_flat - mean) / std 

#     # Node features and PyG Data
#     batched_graph_data = Data(x=point_clouds_flat, edge_index=edge_index, batch=batch_indices)

#     return batched_graph_data


def point_cloud_to_graph(point_clouds, method="knn", k=32, radius=1.5, z_min=0, z_max=20.0, normalize=False):
    """
    Converts a batch of point clouds (batch, N, 3) into a graph representation.
    
    Parameters:
    - point_clouds: Tensor of shape (batch, N, 3), where batch is the number of point clouds.
    - method: "knn" or "radius" for graph construction.
    - k: Number of neighbors for KNN graph.
    - radius: Radius for radius graph.
    - z_min, z_max: Thresholds to filter points based on z-values.
    
    Returns:
    - edge_index: Connectivity information for graph (2, num_edges).
    - normalized_pc: Normalized point cloud tensor.
    - batch: Tensor indicating which points belong to which batch element.
    """

    batch_size, num_points, dim = point_clouds.shape  # (B, N, 3)

    # Reshape into (B*N, 3) to handle it like a flat list
    point_clouds = point_clouds.view(-1, dim)

    # Create batch tensor (Assign each point to its corresponding batch)
    batch = torch.arange(batch_size, device=point_clouds.device).repeat_interleave(num_points)

    # # 1. **Filter out points based on z values**
    # z_values = point_clouds[:, 2]  # Extract z-coordinates
    # mask = (z_values >= z_min) & (z_values <= z_max)

    # counts_per_batch = scatter_add(mask.long(), batch, dim=0)  # shape [batch_size]

    # # Identify which batches have zero remaining points
    # lost_batches = (counts_per_batch == 0)
    # alternate_mask = z_values>0
    # lost_mask = lost_batches[batch]
    # final_mask = mask.clone()
    # final_mask[lost_mask] = alternate_mask[lost_mask]

    # point_clouds = point_clouds[final_mask]
    # batch = batch[final_mask]  # Keep only the valid batch indices

    # 3. **Graph Construction**
    if method == "knn":
        edge_index = knn_graph(point_clouds, k=k, batch=batch)
    elif method == "radius":
        edge_index = radius_graph(point_clouds, r=radius, batch=batch, max_num_neighbors=k)
    else:
        raise ValueError("Invalid method. Choose 'knn' or 'radius'.")

    # 2. **Normalize the filtered point cloud batch-wise (Min-Max Normalization)**
    if normalize:
        min_xyz = scatter_min(point_clouds, batch, dim=0)[0]
        max_xyz = scatter_max(point_clouds, batch, dim=0)[0]
        
        point_clouds = (point_clouds - min_xyz[batch]) / (max_xyz[batch] - min_xyz[batch] + 1e-6)

    return edge_index, point_clouds, batch


class GraphConv(nn.Module):
    """Graph Convolution layer that updates vertex features using edge features"""
    def __init__(self, in_features, out_features):
        super(GraphConv, self).__init__()
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

        # **Normalize the Point Cloud (Zero Mean, Unit Variance)**
        # mean = vertex_features.mean(dim=0, keepdim=True)
        # std = vertex_features.std(dim=0, keepdim=True) + 1e-6
        # vertex_features = (vertex_features - mean) / std  # Apply normalization
        # print(torch.max(vertex_features[:,0]),torch.max(vertex_features[:,1]),torch.max(vertex_features[:,2]))
        
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
            self.layers.append(GraphConv(input_dim, hidden_dim))
            for _ in range(num_layers-2):
                self.layers.append(GraphConv(hidden_dim,2*hidden_dim))
                hidden_dim = 2*hidden_dim
            self.layers.append(GraphConv(hidden_dim,output_dim))

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
        # n_state_inputs = observation_space['vec'].shape[0]
        # self.state_net = nn.Sequential(
        #     nn.Linear(n_state_inputs, 64),
        #     nn.ReLU(),
        #     # nn.Dropout(0.3),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     # nn.Dropout(0.3)
        # )

        # # Combine the CNN and state outputs
        # self.combined_fc = nn.Sequential(
        #     nn.Linear(output_dim + 64, features_dim),
        #     nn.ReLU(),
        #     nn.Linear(features_dim, features_dim),
        #     nn.ReLU()
        # )

    def forward(self, observations):
        # convert point cloud to graph

        edge_index, vertex_features, batch = point_cloud_to_graph(observations['pc'], method='knn', k=4, normalize=False)

        pc_features = self.feature_extractor(edge_index, vertex_features, batch)

        # Process the vector observations
        # vector_features = self.state_net(observations['vec'])

        # combined_features = torch.cat((pc_features, vector_features), dim=1)

        # return self.combined_fc(combined_features)
        return pc_features
    
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




class GSageFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GSageFeatureExtractor, self).__init__()
        # Define GCN layers
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, hidden_dim)
        self.conv3 = SAGEConv(hidden_dim, output_dim)

        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = global_mean_pool

    def forward(self, data):

        # Move batch to same device as the model
        # device = next(self.parameters()).device
        # data = data.to(device)

        # 'data' is a pytorch geometric data object with x, edge_index and batch
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Pass through the layers
        x = self.relu(self.conv1(x, edge_index))
        x = normalize(x, p=2, dim=-1)
        x = self.relu(self.conv2(x, edge_index))
        x = normalize(x, p=2, dim=-1)
        x = self.conv3(x, edge_index)
        x= normalize(x, p=2, dim=-1)

        # Global pooling to get a fixed size representation
        x = self.pool(x, batch)

        return x
    
class GSageFeatureExtractorWrapper(BaseFeaturesExtractor):
    def __init__(self, observation_space, input_dim = 3, hidden_dim = 64, output_dim = 128):
        super(GSageFeatureExtractorWrapper, self).__init__(observation_space, features_dim=output_dim)

        self.feature_extractor = GSageFeatureExtractor(input_dim, hidden_dim, output_dim)

    def forward(self, observations):
        # convert point cloud to graph
        batched_data = point_cloud_to_graph(observations['pc'], radius=0.5)

        # del observations["pc"]

        features = self.feature_extractor(batched_data)
        
        # del batched_data
        # torch.cuda.empty_cache()

        return features


# class GSageFeatureExtractorWrapper(BaseFeaturesExtractor):
#     def __init__(self, observation_space, input_dim = 3, hidden_dim = 64, output_dim = 128, velocity_dim = 2):
#         super(GSageFeatureExtractorWrapper, self).__init__(observation_space, features_dim=output_dim+velocity_dim)

#         self.feature_extractor = GSageFeatureExtractor(input_dim, hidden_dim, output_dim)

#     def forward(self, observations):
#         # convert point cloud to graph
#         batched_data = point_cloud_to_graph(observations['pc'], radius=1)

#         del observations["pc"]

#         features = self.feature_extractor(batched_data)
        
#         del batched_data
#         torch.cuda.empty_cache()

#         velocity = observations["vec"]
#         # velocity = velocity.unsqueeze(-1)
#         combined_features = torch.cat([features, velocity], dim=-1)

#         return combined_features

