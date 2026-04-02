"""
Filename: export_jit.py
Executes the TorchScript tracing pipeline for hierarchical RL.
"""
import torch
import torch.nn as nn
import os

# 1. Reconstruct the exact MLP architecture of the Low-Level Actor
class MinimalActor(nn.Module):
    def __init__(self, obs_dim=48, action_dim=12):
        super().__init__()
        # Matches RSL-RL default Actor structure (256 -> 128 -> 64, ELU)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ELU(),
            nn.Linear(256, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.actor(x)

def main():
    base_dir = "/workspace/isaaclab/isaaclab_master/models"
    ckpt_path = os.path.join(base_dir, "model_38000.pt")
    out_path = os.path.join(base_dir, "go2_low_level_locomotion.pt")
    
    print(f"[INFO] Loading RSL-RL checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    
    # 2. Extract strictly the deterministic actor weights (Ignore Critic and StdDev)
    actor_state_dict = {}
    for key, tensor in state_dict.items():
        if key.startswith("actor.") and "std" not in key:
            actor_state_dict[key] = tensor
            
    # 3. Instantiate and populate the minimal graph
    policy = MinimalActor()
    policy.load_state_dict(actor_state_dict, strict=True)
    policy.eval() # Crucial: Disables dropout/batchnorm for deterministic inference
    
    # 4. Compile via JIT Tracing
    print("[INFO] Tracing computational graph...")
    dummy_obs = torch.zeros(1, 48) # Batch size 1, Obs Dim 48
    with torch.no_grad():
        traced_graph = torch.jit.trace(policy, dummy_obs)
        
    # 5. Serialize for the LibTorch C++ Backend
    traced_graph.save(out_path)
    print(f"[SUCCESS] Zero-copy TorchScript policy exported to: {out_path}")

if __name__ == "__main__":
    main()