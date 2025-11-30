import numpy as np
import os

# ================= CONFIGURATION =================
INPUT_FILE = "./file/expert_data/expert_metadrive_500k_1000scenarios.npz"
OUTPUT_FILE = "./file/expert_data/expert_metadrive_500k_1000scenarios_fixed.npz"
# =================================================

def fix_data_dimensions():
    """
    Fix the extra dimensions in the expert data file.
    Converts (N, 1, D) -> (N, D)
    """
    print("="*60)
    print("🔧 FIXING EXPERT DATA DIMENSIONS")
    print("="*60 + "\n")
    
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Error: Input file not found: {INPUT_FILE}")
        return
    
    print(f"📂 Loading data from {INPUT_FILE}...")
    data = np.load(INPUT_FILE, allow_pickle=True)
    
    # Load all fields
    observations = data["observations"]
    actions = data["actions"]
    next_observations = data["next_observations"]
    rewards = data["rewards"]
    dones = data["dones"]
    
    print(f"\n📊 ORIGINAL SHAPES:")
    print(f"   Observations:      {observations.shape}")
    print(f"   Actions:           {actions.shape}")
    print(f"   Next Observations: {next_observations.shape}")
    print(f"   Rewards:           {rewards.shape}")
    print(f"   Dones:             {dones.shape}")
    
    # Fix dimensions by squeezing axis=1
    observations_fixed = observations.squeeze(axis=1)
    actions_fixed = actions.squeeze(axis=1)
    next_observations_fixed = next_observations.squeeze(axis=1)
    rewards_fixed = rewards.squeeze(axis=1) if rewards.ndim > 1 else rewards
    dones_fixed = dones.squeeze(axis=1) if dones.ndim > 1 else dones
    
    print(f"\n✅ FIXED SHAPES:")
    print(f"   Observations:      {observations_fixed.shape}")
    print(f"   Actions:           {actions_fixed.shape}")
    print(f"   Next Observations: {next_observations_fixed.shape}")
    print(f"   Rewards:           {rewards_fixed.shape}")
    print(f"   Dones:             {dones_fixed.shape}")
    
    # Verify data integrity
    print(f"\n🔍 DATA INTEGRITY CHECK:")
    print(f"   Number of samples: {len(observations_fixed):,}")
    print(f"   Rewards mean:      {rewards_fixed.mean():.4f} (per-step)")
    print(f"   Actions mean:      {actions_fixed.mean(axis=0)}")
    print(f"   Actions std:       {actions_fixed.std(axis=0)}")
    print(f"   Actions range:     [{actions_fixed.min():.3f}, {actions_fixed.max():.3f}]")
    
    # Check for NaN/Inf
    has_nan = (np.isnan(observations_fixed).any() or 
               np.isnan(actions_fixed).any() or
               np.isnan(next_observations_fixed).any())
    has_inf = (np.isinf(observations_fixed).any() or 
               np.isinf(actions_fixed).any() or
               np.isinf(next_observations_fixed).any())
    
    if has_nan or has_inf:
        print(f"\n   ⚠️  WARNING: Data contains NaN or Inf values!")
        print(f"      NaN: {has_nan}, Inf: {has_inf}")
    else:
        print(f"\n   ✅ No NaN or Inf values detected")
    
    # Save fixed data
    print(f"\n💾 Saving fixed data to {OUTPUT_FILE}...")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    np.savez_compressed(
        OUTPUT_FILE,
        observations=observations_fixed,
        actions=actions_fixed,
        next_observations=next_observations_fixed,
        rewards=rewards_fixed,
        dones=dones_fixed
    )
    
    file_size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
    print(f"✅ Fixed data saved! File size: {file_size_mb:.2f} MB")
    
    print(f"\n{'='*60}")
    print("🎉 DATA FIX COMPLETE!")
    print("="*60)
    print(f"\n📝 NEXT STEPS:")
    print(f"   1. Update train_bc_imitation.py line 21:")
    print(f'      EXPERT_DATA_PATH = "{OUTPUT_FILE}"')
    print(f"\n   2. Re-train BC:")
    print(f"      python train_bc_imitation.py")
    print(f"\n   3. The BC should now work correctly!")
    print("="*60 + "\n")

if __name__ == "__main__":
    fix_data_dimensions()