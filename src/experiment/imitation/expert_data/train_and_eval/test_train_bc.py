import os
import json
import numpy as np
import torch
import gymnasium as gym
from metadrive.envs.metadrive_env import MetaDriveEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.policies import ActorCriticPolicy
import imitation.algorithms.bc as bc
from imitation.data.types import Transitions
from sklearn.model_selection import train_test_split
import functools
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# ================= CONFIGURATION =================
EXPERT_DATA_PATH = "./file/expert_data/expert_metadrive_500k_noisy_v5.npz"
MODEL_SAVE_PATH = "./file/model/bc_256.zip"
STATS_PATH = "./file/expert_data/expert_metadrive_500k_1200eps_normalized_stats.pkl"
BEST_POLICY_PATH = "./file/model/bc_256_policy_best.zip"
TRAFFIC_DENSITY = 0.3  # Must match collection density
SEED = 42
BATCH_SIZE = 64
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ALGO_NAME = "PPO"  # or "LeakyPPO"
# =================================================

def load_rl_architecture(algo_name="PPO"):
    """Load the same architecture that RL will use."""
    if "LeakyPPO" in algo_name:
        filename = "leaky_ppo_metadrive_optuna_1.5M_best_params.json"
    else:
        filename = "ppo_metadrive_optuna_1.5M_best_params.json"
    
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found, defaulting to medium")
        return [256, 256]
    
    with open(filename, "r") as f:
        params = json.load(f)
    
    # Map architecture strings to actual configs
    arch_map = {
        "wide":   [400, 300],
        "medium": [256, 256],
        "deep":   [256, 256, 256]
    }
    
    arch_key = params.get("net_arch", "medium")
    architecture = arch_map.get(arch_key, [256, 256])
    
    print(f"Using RL-matched architecture: {arch_key} -> {architecture}")
    return architecture

def load_expert_transitions(path):
    """
    Loads .npz data and converts it to imitation's Transitions format.
    """
    print(f"Loading data from {path}...")
    data = np.load(path)
    
    # Extract arrays
    obs = data['observations']
    acts = data['actions']
    next_obs = data['next_observations']
    dones = data['dones']
    
    dones = data['dones'].astype(bool)

    # imitation library expects an 'infos' list (can be empty dicts)
    # This is required for the Transitions data structure
    infos = [{}] * len(obs)
    
    print(f"✓ Loaded {len(obs)} transitions")
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Action shape: {acts.shape}")

    return Transitions(
        obs=obs,
        acts=acts,
        infos=infos,
        next_obs=next_obs,
        dones=dones
    )

def train():
    # Setup Environment (Required for Action/Obs spaces)
    # We use the same config as collection to ensure space consistency
    env_config = {
        "use_render": False,
        "traffic_density": TRAFFIC_DENSITY,
        "num_scenarios": 100,
        "start_seed": 1000,
    }
    env = MetaDriveEnv(env_config)
    
    # Load Data
    transitions = load_expert_transitions(EXPERT_DATA_PATH)

    # Add this before training
    print(f"Obs Mean: {transitions.obs.mean():.3f}")
    print(f"Obs Max:  {transitions.obs.max():.3f}")
    print(f"Obs Min:  {transitions.obs.min():.3f}")
    print(f"Acts Mean: {transitions.acts.mean():.3f}")
    print(f"Acts Max:  {transitions.acts.max():.3f}")
    print(f"Acts Min:  {transitions.acts.min():.3f}")

    # ==== DATA SPLIT ====
    # Split data into training and validation sets
    total_samples = len(transitions)
    indices = np.arange(total_samples)
    train_idx, val_idx = train_test_split(indices, test_size=0.1, random_state=42)

    print(f"\nData Split:")
    print(f"   Training on:   {len(train_idx):,} steps")
    print(f"   Validating on: {len(val_idx):,} steps")
    
    train_transitions = Transitions(
        obs=transitions.obs[train_idx],
        acts=transitions.acts[train_idx],
        infos=[{}] * len(train_idx),
        next_obs=transitions.next_obs[train_idx],
        dones=transitions.dones[train_idx]
    )
    
    val_transitions = Transitions(
        obs=transitions.obs[val_idx],
        acts=transitions.acts[val_idx],
        infos=[{}] * len(val_idx),
        next_obs=transitions.next_obs[val_idx],
        dones=transitions.dones[val_idx]
    )
    # ==== END DATA SPLIT ====
    
    # Load matching architecture
    net_arch = load_rl_architecture(ALGO_NAME)  # or "LeakyPPO" depending on what you're warmstarting

    # Initialize BC Trainer
    # We use a standard FeedForward policy (MlpPolicy)
    rng = np.random.default_rng(SEED)
    
    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=train_transitions,
        rng=rng,
        batch_size=BATCH_SIZE,
        device=DEVICE,
        # Use a policy that outputs Tanh (since MetaDrive actions are -1 to 1)
        policy=ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            lr_schedule=lambda _: 1e-4,
            net_arch=net_arch,
        )
    )
    
    # Convert validation data to tensors ONCE to avoid re-doing it every epoch
    train_obs_tensor = torch.as_tensor(train_transitions.obs).to(DEVICE)
    train_acts_tensor = torch.as_tensor(train_transitions.acts).to(DEVICE)
    val_obs_tensor = torch.as_tensor(val_transitions.obs).to(DEVICE)
    val_acts_tensor = torch.as_tensor(val_transitions.acts).to(DEVICE)

    best_val_mse = float('inf')
    patience = 0
    MAX_PATIENCE = 10

    print("\nStarting BC Training...")
    print("=" * 70)

    # Train
    for i in range(EPOCHS):
        # Train for one epoch
        # (There is no simple 'train_epoch' in imitation, so we use n_batches)
        # n_batches = n_samples // batch_size
        loss_dict = bc_trainer.train(n_epochs=1) 
        
        # Calculate Validation Loss (MSE) manually for clarity
        # We predict actions for validation obs
        with torch.no_grad():
            # NLL (what BC optimizes)
            dist = bc_trainer.policy.get_distribution(val_obs_tensor)
            log_prob = dist.log_prob(val_acts_tensor)
            val_nll = -log_prob.mean().item()
            
            # MSE (interpretable)
            val_pred, _ = bc_trainer.policy.predict(val_obs_tensor, deterministic=True)
            val_pred = torch.as_tensor(val_pred).to(DEVICE)
            val_mse = torch.nn.functional.mse_loss(val_pred, val_acts_tensor).item()
            
            # Per-action MSE
            steering_mse = torch.nn.functional.mse_loss(val_pred[:, 0], val_acts_tensor[:, 0]).item()
            throttle_mse = torch.nn.functional.mse_loss(val_pred[:, 1], val_acts_tensor[:, 1]).item()

        # Print metrics
        print(f"Epoch {i+1:3d}/{EPOCHS} | "
            f"Val NLL: {val_nll:.4f} | "
            f"Val MSE: {val_mse:.4f} | "
            f"Steer: {steering_mse:.4f} | "
            f"Throt: {throttle_mse:.4f}")
        
        # Early stopping based on MSE
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            bc_trainer.policy.save(BEST_POLICY_PATH)
            patience = 0
        else:
            patience += 1
            if patience >= MAX_PATIENCE:
                print(f"\n⚠️  Early stopping triggered (no improvement for {MAX_PATIENCE} epochs)")
                break
    
    print("=" * 70)
    print(f"✅ Training Complete | Best Val MSE: {best_val_mse:.4f}")
        
    # RELOAD the best checkpoint from disk
    print("\nLoading best model for evaluation...")
    # bc_trainer.policy = ActorCriticPolicy.load(BEST_POLICY_PATH)

    # Create a custom loader that forces weights_only=False
    custom_load = functools.partial(torch.load, weights_only=False)
    
    # Swap, Load, Restore
    original_load = torch.load
    torch.load = custom_load
    try:
        best_policy_loaded = ActorCriticPolicy.load(BEST_POLICY_PATH)
        bc_trainer.policy.load_state_dict(best_policy_loaded.state_dict())
        print("✓ Best policy weights loaded successfully.")
    finally:
        torch.load = original_load

    
    # 5. Evaluate
    print("Evaluating Policy in environment...")

    # Create evaluation environment with VecNormalize
    eval_env = DummyVecEnv([lambda: MetaDriveEnv(env_config)])

    # Load the normalization stats from data collection
    stats_path = STATS_PATH
    if os.path.exists(stats_path):
        eval_env = VecNormalize.load(stats_path, eval_env)
        eval_env.training = False  # Don't update stats during eval
        eval_env.norm_reward = False  # Return raw rewards
        print(f"  Loaded VecNormalize stats from {stats_path}")
    else:
        print(f"  WARNING: Stats file not found at {stats_path}")
        print(f"  Evaluation results will be incorrect!")
    
    mean_reward, std_reward = evaluate_policy(bc_trainer.policy, eval_env, n_eval_episodes=10)
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        
    # 6. Save the trained policy
    bc_trainer.policy.save(MODEL_SAVE_PATH)
    print(f"Policy saved to {MODEL_SAVE_PATH}")

    print(f"\n{'='*60}")
    print("BC POLICY VERIFICATION")
    print(f"{'='*60}")
    param_count = sum(p.numel() for p in bc_trainer.policy.parameters())
    print(f"Total parameters: {param_count:,}")
    print(f"Architecture: {net_arch}")
    print(f"This must match RL training architecture!")
    print(f"{'='*60}\n")

    eval_env.close()
    env.close()

if __name__ == "__main__":
    train()