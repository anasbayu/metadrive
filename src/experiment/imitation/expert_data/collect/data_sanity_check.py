import numpy as np

DATA_PATH = "./file/expert_data/expert_metadrive_500k_noisy_v5.npz" 

try:
    data = np.load(DATA_PATH)
    actions = data['actions']
    obs = data['observations']
    
    print(f"Loaded {len(actions)} steps.")
    print("-" * 30)
    print(f"Action Min: {actions.min():.4f}")
    print(f"Action Max: {actions.max():.4f}")
    print("-" * 30)
    print(f"Obs Min:    {obs.min():.4f}")
    print(f"Obs Max:    {obs.max():.4f}")
    print("-" * 30)

    # The Pass/Fail Criteria
    if actions.max() > 1.05 or actions.min() < -1.05:
        print("❌ FAILED: Actions are out of bounds! Do not train.")
    elif np.isnan(actions).any():
        print("❌ FAILED: Actions contain NaNs!")
    else:
        print("✅ PASSED: Data is clean and ready for training.")
        
except Exception as e:
    print(f"Waiting for file... ({e})")