import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os

"""
PyTorch BC Training with SB3-Compatible Architecture
This ensures easy warmstart for PPO!
"""

# ============== CONFIGURATIONS =============
EXPERT_DATA_PATH = "./file/expert_data/expert_metadrive_500k_1000scenarios_fixed.npz"
BC_POLICY_SAVE_PATH = "./file/model/bc_policy_pytorch_sb3_compatible_new.zip"
TRAFFIC_DENSITY = 0.0

BC_HYPERPARAMS = {
    "n_epochs": 50,  # Based on your working PyTorch approach
    "batch_size": 256,
    "learning_rate": 3e-4,
}
# =================================================

def train_bc_pytorch_sb3_style():
    """
    Train BC using PyTorch but with SB3's ActorCriticPolicy architecture
    This makes PPO warmstart trivial!
    """
    print("="*60)
    print("🚗 PYTORCH BC TRAINING (SB3-Compatible)")
    print("="*60)
    print(f"Epochs: {BC_HYPERPARAMS['n_epochs']}")
    print(f"Batch Size: {BC_HYPERPARAMS['batch_size']}")
    print("="*60 + "\n")
    
    # 1. Load data
    print("1. Loading expert data...")
    data = np.load(EXPERT_DATA_PATH)
    obs = torch.FloatTensor(data["observations"])
    actions = torch.FloatTensor(data["actions"])
    
    print(f"   Observations: {obs.shape}")
    print(f"   Actions: {actions.shape}")
    
    # 2. Create environment to get spaces
    print("\n2. Creating environment for architecture...")
    temp_env = MetaDriveEnv({
        "use_render": False,
        "log_level": 50,
        "traffic_density": TRAFFIC_DENSITY,
    })
    
    # 3. Create ActorCriticPolicy (SAME as PPO uses!)
    print("\n3. Creating ActorCriticPolicy (SB3-compatible)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    policy = ActorCriticPolicy(
        observation_space=temp_env.observation_space,
        action_space=temp_env.action_space,
        lr_schedule=lambda _: BC_HYPERPARAMS["learning_rate"],
        net_arch=dict(pi=[256, 256], vf=[256, 256])  # Same as PPO
    ).to(device)
    
    temp_env.close()
    
    print(f"   Device: {device}")
    print(f"   Architecture: pi=[256, 256], vf=[256, 256]")
    
    # 4. Setup training
    print("\n4. Setting up training...")
    dataset = TensorDataset(obs, actions)
    dataloader = DataLoader(dataset, batch_size=BC_HYPERPARAMS["batch_size"], shuffle=True)
    
    optimizer = torch.optim.Adam(policy.parameters(), lr=BC_HYPERPARAMS["learning_rate"])
    criterion = nn.MSELoss()
    
    # 5. Train (only policy network, ignore value network)
    print("\n5. Training BC...")
    print("="*60)
    
    policy.train()
    for epoch in range(BC_HYPERPARAMS["n_epochs"]):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            
            # Forward pass through policy
            # Get latent features
            features = policy.extract_features(batch_obs)
            latent_pi = policy.mlp_extractor.forward_actor(features)
            
            # Get action predictions
            actions_pred = policy.action_net(latent_pi)
            
            # Compute loss
            loss = criterion(actions_pred, batch_actions)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{BC_HYPERPARAMS['n_epochs']}: Loss = {avg_loss:.6f}")
    
    print("\n✅ Training complete!")
    
    # 6. Save policy and normalization stats
    print(f"\n6. Saving policy to {BC_POLICY_SAVE_PATH}...")
    os.makedirs(os.path.dirname(BC_POLICY_SAVE_PATH), exist_ok=True)
    
    # Save normalization stats
    norm_stats_path = BC_POLICY_SAVE_PATH.replace('.zip', '_norm_stats.npz')
    obs_mean = obs.mean(dim=0).cpu().numpy()
    obs_std = obs.std(dim=0).cpu().numpy() + 1e-8
    np.savez(norm_stats_path, mean=obs_mean, std=obs_std)
    print(f"   Saved normalization stats to {norm_stats_path}")
    
    policy.save(BC_POLICY_SAVE_PATH)
    print("✅ Policy saved!")
    
    # 7. Quick evaluation
    print("\n7. Quick evaluation...")
    
    # Calculate normalization stats from training data
    print("   Computing observation normalization from training data...")
    obs_mean = obs.mean(dim=0).cpu().numpy()
    obs_std = obs.std(dim=0).cpu().numpy() + 1e-8  # Add epsilon to avoid division by zero
    
    env = MetaDriveEnv({
        "use_render": False,
        "manual_control": False,
        "log_level": 50,
        "num_scenarios": 10,
        "start_seed": 0,
        "traffic_density": TRAFFIC_DENSITY,
    })
    
    policy.eval()
    successes = 0
    
    for ep in range(10):
        obs_raw, _ = env.reset()
        terminated = False
        truncated = False
        
        while not terminated and not truncated:
            # Normalize observation before prediction
            obs_normalized = (obs_raw - obs_mean) / obs_std
            action, _ = policy.predict(obs_normalized, deterministic=True)
            obs_raw, _, terminated, truncated, info = env.step(action)
        
        if info.get("arrive_dest", False):
            successes += 1
    
    env.close()
    
    print(f"   Quick test: {successes}/10 success ({successes*10}%)")
    
    print("\n" + "="*60)
    print("🎉 BC TRAINING COMPLETE")
    print("="*60)
    print(f"\n✅ Model saved: {BC_POLICY_SAVE_PATH}")
    print(f"✅ Architecture: SB3 ActorCriticPolicy (compatible with PPO)")
    print(f"✅ Ready for PPO warmstart!")
    print("\n📝 NEXT STEP:")
    print(f"   Use this model for PPO warmstart in train_ppo_imitation.py")
    print("="*60 + "\n")

if __name__ == "__main__":
    train_bc_pytorch_sb3_style()