import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# ================= CONFIGURATION =================
# Make sure this points to your latest BRAKE-INJECTED dataset
INPUT_DATA = "./file/expert_data/expert_metadrive_500k_1200eps_with_recovery_v3.npz" 
MODEL_SAVE_PATH = "./file/models/bc_agent_forced2d.pth"
STATS_SAVE_PATH = "./file/models/normalization_stats.npz"

BATCH_SIZE = 64
EPOCHS = 30           
LEARNING_RATE = 3e-4 
# =================================================

class BCAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(BCAgent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        return self.net(x)

def train_bc():
    os.makedirs("./models", exist_ok=True)
    
    print(f"Loading data from {INPUT_DATA}...")
    if not os.path.exists(INPUT_DATA):
        print("❌ Data file not found!")
        return

    data = np.load(INPUT_DATA, allow_pickle=True)
    obs = data['observations']
    actions = data['actions']
    
    # === CRITICAL FIX: SQUEEZE OBSERVATIONS ===
    # If shape is (N, 1, 259) -> Convert to (N, 259)
    if obs.ndim == 3:
        print(f"  Detected Obs Shape {obs.shape}. Squeezing...")
        obs = obs.squeeze(axis=1)

    # If shape is (N, 1, 2) -> Convert to (N, 2)
    if actions.ndim == 3:
        print(f"  Detected Act Shape {actions.shape}. Squeezing...")
        actions = actions.squeeze(axis=1)
    # ==========================================

    print(f"Training on {len(obs)} samples.")
    print(f"Observation Shape: {obs.shape}") # Should be (N, 259)
    print(f"Action Shape: {actions.shape}")   # Should be (N, 2)

    # 1. NORMALIZE OBSERVATIONS
    print("Calculating normalization statistics...")
    obs_mean = obs.mean(axis=0)
    obs_std = obs.std(axis=0) + 1e-8 
    
    os.makedirs(os.path.dirname(STATS_SAVE_PATH), exist_ok=True) 
    np.savez(STATS_SAVE_PATH, mean=obs_mean, std=obs_std)
    print(f"✅ Normalization stats saved to {STATS_SAVE_PATH}")
    
    normalized_obs = (obs - obs_mean) / obs_std

    # 2. Convert to PyTorch
    tensor_obs = torch.FloatTensor(normalized_obs)
    tensor_act = torch.FloatTensor(actions)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training device: {device}")

    dataset = TensorDataset(tensor_obs, tensor_act)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 3. Setup Model
    # Now obs.shape[1] will correctly be 259 (not 1)
    model = BCAgent(obs.shape[1], actions.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss() 

    # 4. Train
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for b_obs, b_act in dataloader:
            b_obs, b_act = b_obs.to(device), b_act.to(device)
            
            optimizer.zero_grad()
            pred_act = model(b_obs)
            loss = criterion(pred_act, b_act)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"🎉 Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train_bc()