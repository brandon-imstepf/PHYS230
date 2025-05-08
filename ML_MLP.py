# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# --- Set number of epochs for training --- #
num_epochs = 1000000

# --- Set & Create Directory --- #
# Set the main directory:
main_dir = r"C:\Users\brand\Desktop\School\UCM\UCM Spring 2025\PHYS 230\Project"
os.chdir(main_dir)
# Create directories for saving data and visualizations:
vis_dir = f"ML/num_epochs_{num_epochs}"
os.makedirs(vis_dir, exist_ok=True)

# --- Load and Prepare Data --- #
data = pd.read_csv("Data/Main Info/parameter_pattern_log.csv")
X = data[["F", "k"]].values
y = data["Pattern"].astype(float).values.reshape(-1, 1)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)




# --- Define Neural Network ---
class PatternClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

model = PatternClassifier()
loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


# --- Training Loop ---
train_losses = []
val_losses = []
train_acc = []
val_acc = []

for epoch in range(num_epochs):
    # Training
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    train_losses.append(loss.item())

    with torch.no_grad():
        # Training accuracy
        pred_class = (y_pred > 0.5).float()
        acc = (pred_class == y_train).float().mean().item()
        train_acc.append(acc)

        # Validation
        model.eval()
        y_val_pred = model(X_val)
        val_loss = loss_fn(y_val_pred, y_val)
        val_losses.append(val_loss.item())

        pred_val_class = (y_val_pred > 0.5).float()
        val_accuracy = (pred_val_class == y_val).float().mean().item()
        val_acc.append(val_accuracy)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train Loss={loss.item():.4f}, Val Loss={val_loss.item():.4f}, Val Acc={val_accuracy:.4f}")




# --- Plot Metrics ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss over Epochs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_acc, label="Train Acc")
plt.plot(val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Accuracy over Epochs")
plt.legend()
plt.tight_layout()
plt.savefig(f"{vis_dir}/training_curves.png")
plt.close()

def plot_only_pattern_map(F_vals, k_vals, pattern_array,directory,name,bounds,color_map):
    """
    Plot pattern map using imshow if data is full square grid, or scatter if partial (e.g., train/val/test).
    
    Parameters:
    - pattern_array: Array-like, pattern boolean values (True/False or 1/0)
    - F_vals, k_vals: If using scatter, must pass corresponding F and k coordinate arrays.
    - title: Plot title
    - cmap: Colormap to use
    """
    pattern_array = np.array(pattern_array).astype(float)  # Ensure numeric
    N = len(pattern_array)
    grid_size = int(np.sqrt(N))

    plt.figure(figsize=(6, 5))

    if grid_size * grid_size == N and F_vals is None and k_vals is None:
        # Use imshow for full grid
        Z = pattern_array.reshape((grid_size, grid_size))
        plt.imshow(Z, origin='lower', extent=[0, 1, 0, 1], aspect='auto', cmap=color_map)
    else:
        # Use scatter for partial data
        assert F_vals is not None and k_vals is not None, "F_vals and k_vals required for scatter plot."
        plt.scatter(F_vals, k_vals, c=pattern_array, cmap=color_map, alpha=0.7, edgecolor='k', s=10, marker='s')

    plt.xlabel("F")
    plt.ylabel("k")
    plt.title(name)
    plt.tight_layout()
    plt.xlim(bounds[0], bounds[1])
    plt.ylim(bounds[0], bounds[1])
    os.makedirs(directory, exist_ok=True)
    plt.savefig(f"{directory}/pattern_map_{name}.png"); plt.close()


# --- Plot Pattern Map for train, test and val in different colors --- #
plt.figure(figsize=(8, 6))

# Training set
plt.scatter(X_train[:, 0].numpy(), X_train[:, 1].numpy(), c='blue', marker='s', label="Train", alpha=0.5)

# Validation set
plt.scatter(X_val[:, 0].numpy(), X_val[:, 1].numpy(), c='purple', marker='s', label="Validation", alpha=1)

# Test set
plt.scatter(X_test[:, 0].numpy(), X_test[:, 1].numpy(), c='red', marker='s', label="Test", alpha=0.5)

plt.xlabel("k (Kill)")
plt.ylabel("F (Feed)")
plt.title("Pattern Map with Train, Validation, and Test Sets")
plt.legend()
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.savefig(f"{vis_dir}/pattern_map_combined.png")
plt.close()

bounds = [0,1]


plot_only_pattern_map(data["F"].values, data["k"].values, data["Pattern"].values, vis_dir, "Whole Dataset",bounds,'Greens')
plot_only_pattern_map(X_train[:, 0].numpy(), X_train[:, 1].numpy(), y_train.numpy(), vis_dir, "Training Dataset", bounds, 'Blues')
plot_only_pattern_map(X_val[:, 0].numpy(), X_val[:, 1].numpy(), y_val.numpy(), vis_dir, "Validation Dataset", bounds, 'Purples')
plot_only_pattern_map(X_test[:, 0].numpy(), X_test[:, 1].numpy(), y_test.numpy(), vis_dir, "Testing Dataset", bounds, 'Reds')



# --- Evaluate on Test Set ---
model.eval()
with torch.no_grad():
    y_test_pred = model(X_test)
    test_pred_class = (y_test_pred > 0.5).float()
    test_accuracy = (test_pred_class == y_test).float().mean().item()
    print(f"Test Accuracy: {test_accuracy:.4f}")


    # --- Generate Predictions and Plot --- #
    with torch.no_grad():
        # Generate predictions for the entire dataset
        X_full = torch.tensor(scaler.transform(data[["F", "k"]].values), dtype=torch.float32)
        y_full_pred = model(X_full)
        y_full_pred_class = (y_full_pred > 0.5).float().numpy()

    # Plot the predicted pattern map
    plot_only_pattern_map(
        data["F"].values,
        data["k"].values,
        y_full_pred_class.flatten(),
        vis_dir,
        "Predicted Pattern Map",
        bounds,
        'Oranges'
    )