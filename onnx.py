import os
import torch
import argparse
from stable_baselines3 import SAC, PPO
# from sb3_contrib import RecurrentPPO
ext_list = ["pi_features_extractor", "mlp_extractor.policy_net", "action_net"]
"""
pi_features_extractor:
    Input: Depth image format -> (1,128,128,1)
    output: 128 features ---- check for the shape of the output
mlp_extractor.policy_net:
    input: pi_features_extractor_output --> 128
    output: 128 ----> check for the shape of the output
action_net:
    input: mlp_extractor.policy_net --> 128
    output: 2 ----> check for the shape of the output
"""
class DictInputWrapper(torch.nn.Module):
    def __init__(self, policy):
        super(DictInputWrapper, self).__init__()
        self.policy = policy
    def forward(self, depth):
        # Recreate the observation dictionary
        obs = {'img': depth}
        # Forward pass through the actual policy using dict observation
        return self.policy(obs)
def export_to_onnx(model_part, input_tensor, output_path, input_names=None, output_names=None):
    # Use the provided input name or default to "input"
    input_name = input_names[0] if input_names else "input"
    output_name = output_names[0] if output_names else "output"
    torch.onnx.export(
        model_part,
        input_tensor,
        output_path,
        export_params=True,
        opset_version=11,
        input_names=input_names if input_names else ["input"],
        output_names=output_names if output_names else ["output"],
        dynamic_axes={"input_name": {0: 'batch_size'}}
    )
    print(f"Exported {output_path} to ONNX successfully!")
def save_model_parts_as_onnx(model, output_dir="onnx_models"):
    os.makedirs(output_dir, exist_ok=True)
    model.policy.eval()
    # Extract CNN Network
    print("Extracting and converting CNN_Depth Network...")
    sample_cnn = torch.randn(1, 6, 128, 128)  # Adjust this based on actual expected input shape
    dummy_input = {
        'img': torch.randn(1, *model.observation_space['img'].shape).to(torch.device('cuda'))
    }
    # Wrap the policy in the custom input wrapper
    wrapped_policy = DictInputWrapper(model.policy.pi_features_extractor)
    export_to_onnx(
        wrapped_policy,
        (dummy_input['img']),
        os.path.join(output_dir, "CNN_depth_new_3.onnx"),
        input_names=["input"],
        output_names=["output"]
    )
    # Extract MLP Network
    print("Extracting and converting MLP extractor Network...")
    sample_cnn = torch.randn(1,128).to(torch.device('cuda'))  # Adjust this based on actual expected input shape
    export_to_onnx(
        model.policy.mlp_extractor.policy_net,
        sample_cnn,
        os.path.join(output_dir, "MLP_extractor_new_3.onnx"),
        input_names=["input"],
        output_names=["output"]
    )
    # Extract action Network
    print("Extracting and converting MLP extractor Network...")
    sample_cnn = torch.randn(1,128).to(torch.device('cuda'))  # Adjust this based on actual expected input shape
    export_to_onnx(
        model.policy.action_net,
        sample_cnn,
        os.path.join(output_dir, "Action_net_new_3.onnx"),
        input_names=["input"],
        output_names=["output"]
    )
def load_model_and_save_parts(model_type, model_path, output_dir):
    # model_class = {"PPO": PPO, "SAC": SAC}[model_type]
    # Load the model from .zip file to ensure weights are correct
    model = PPO.load(model_path)
    print("Model loaded successfully!")
    # Save parts to ONNX
    save_model_parts_as_onnx(model, output_dir)
if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Extract and convert parts of a saved model to ONNX format.")
    # parser.add_argument('--model_type', type=str, choices=["PPO", "SAC", "RecurrentPPO"], help='Type of model to use', default="SAC")
    # parser.add_argument('--model_path', type=str, help='Path to the saved model (.zip) to extract parts from', required=True)
    # parser.add_argument('--output_dir', type=str, help='Directory to save the ONNX files', default='./onnx_models')
    # args = parser.parse_args()
    model_type = PPO
    model_path = "/home/mmkr/Car_LocoTransformer/Mujoco_learning/Pushr_car_simulation/results/models/PPO_Img_300k"
    output_dir = "/home/mmkr/Car_LocoTransformer/Mujoco_learning/Pushr_car_simulation/results/onnx"
    # Execute the extraction and conversion
    # load_model_and_save_parts(args.model_type, args.model_path, args.output_dir)
    load_model_and_save_parts(model_type, model_path, output_dir)