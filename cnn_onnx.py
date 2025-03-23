import torch
import onnx
from stable_baselines3 import SAC

def export_cnn_from_sac_model(model_path, onnx_path):
    # Load the trained SAC model
    model = SAC.load(model_path)
    
    # Extract the CNN (DepthMap Feature Extractor) from the actor's feature extractor
    cnn_extractor = model.policy.actor.features_extractor.cnn
    cnn_extractor.eval()
    
    # Create a dummy depth map input
    dummy_input = torch.randn(1, 1, 64, 128).to(torch.device('cuda'))  # Assuming depth input is (64, 128)
    
    # Export to ONNX
    torch.onnx.export(cnn_extractor,          # Export only the CNN part
                      dummy_input,            # Input tensor
                      onnx_path,              # Output ONNX file
                      export_params=True,     # Store weights
                      opset_version=11,       # ONNX version
                      input_names=['depth_input'],  # Name for input
                      output_names=['cnn_output'],  # Name for output
                      dynamic_axes={'depth_input': {0: 'batch_size'}, 'cnn_output': {0: 'batch_size'}})
    
    print(f"CNN model exported to {onnx_path}")

# Usage
model_path = "/home/mmkr/Car_LocoTransformer/Mujoco_learning/Pushr_car_simulation/ppo_car.zip"
export_cnn_from_sac_model(model_path, './cnn_depth_feature_extractor.onnx')
