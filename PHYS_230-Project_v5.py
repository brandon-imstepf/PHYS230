# --- Imports ---
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio
from tqdm import tqdm
import csv
import shutil
from scipy.fft import fft2, fftshift


## --- INPUTS --- ##

# --- Main Parameter Sweep --- #
F_vals = np.round(np.linspace(0.65, 0.75, 3), 3) # np.round to round to 3 decimal places
k_vals = np.round(np.linspace(0.55, 0.65, 3), 3)
params = {"n": 256, # Size of the matrix, essentially the "resolution" of the image 
          "steps": 10000, # How far into the simulation to go. The further, the more accurate (but not too far!)
          "Du": 0.16, # Diffusion rate of u
          "Dv": 0.08, # Diffusion rate of v
          "dx": 1.0, # Spatial resolution (size of each pixel in the grid)
          "dt": 1.0, # Time step (how far to move the simulation forward each time)
          "snapshot_interval": 100 # How often to take a snapshot of the simulation e.g. how often to run the simulation
          }

# --- Simulation Functions ---
def gray_scott(u, v, Du, Dv, F, k, dx, dt):
    """
    Compute the next timestep of the Gray-Scott system.
    Inputs: u, v = concentration grids
            Du, Dv = diffusion rates
            F, k = feed/kill rates
            dx, dt = spatial and temporal resolution
    Returns: updated u, v
    """
    lap_u = (np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
             np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4*u) / dx**2
    lap_v = (np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) +
             np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4*v) / dx**2
    uvv = u * v * v
    return u + dt * (Du * lap_u - uvv + F * (1 - u)), v + dt * (Dv * lap_v + uvv - (F + k) * v)

def seed_initial_conditions(n):
    """
    Initialize a central square perturbation in u and v.
    Inputs: n = size of the grid
    Returns: u, v = initial concentration grids
    """
    u, v = np.ones((n, n)), np.zeros((n, n))
    r = n // 10
    u[n//2-r:n//2+r, n//2-r:n//2+r] = 0.5
    v[n//2-r:n//2+r, n//2-r:n//2+r] = 0.25
    return u, v

def run_simulation(F, k, params):
    """
    Run simulation and return snapshots of u and v.
    Inputs: F, k = feed and kill rates
            params = dictionary of simulation parameters
    Returns: snapshots of u and v at specified intervals
    """
    u, v = seed_initial_conditions(params["n"])
    su, sv = [], []
    for step in range(params["steps"]):
        u, v = gray_scott(u, v, params["Du"], params["Dv"], F, k, params["dx"], params["dt"])
        if step % params["snapshot_interval"] == 0:
            su.append(u.copy())
            sv.append(v.copy())
    return np.array(su), np.array(sv)

def fft_energy(img,graph_boolean=False):
    """
    Compute the FFT energy of an image and return the total energy outside a central region.
    Inputs:
        img: 2D array representing the image.
    Returns:
        total_energy: Sum of the power spectrum outside the central region.
        power: 2D array of the power spectrum.
    """
    # Compute the 2D FFT of the image and shift the zero-frequency component to the center
    fft_img = fftshift(fft2(img))
    
    # Compute the power spectrum (magnitude squared of the FFT)
    power = np.abs(fft_img)**2
    
    # Define the center of the power spectrum
    center = np.array(power.shape) // 2
    
    # Define a circular region around the center to exclude (low-frequency components)
    r = min(center) // 4  # Radius of the excluded region
    mask = np.ones_like(power, dtype=bool)  # Initialize a mask with all True values
    y, x = np.ogrid[:power.shape[0], :power.shape[1]]  # Create grid indices
    mask[(y - center[0])**2 + (x - center[1])**2 <= r**2] = 0  # Exclude the central region

    if graph_boolean == True:
        plt.imshow(mask, cmap='gray')
        plt.title("Mask for Excluded Region")
        plt.axis("off")
        plt.savefig("mask.png")
        plt.imshow(power, cmap='viridis')
        plt.title("Power Spectrum")
        plt.axis("off")
        plt.savefig("power_spectrum.png")
        plt.close()
    
    # Sum the power spectrum values outside the excluded region
    return np.sum(power[mask]), power

def graph_fft_power_spectrum(img, directory):
    """
    Generate and save a visualization of the FFT power spectrum of an image.
    Inputs:
        img: 2D array representing the image.
        directory: Path to the directory where the plot will be saved.
    """
    # Compute the power spectrum of the image
    power = fft_energy(img)[1]
    
    # Create a plot of the log-scaled power spectrum
    plt.figure(figsize=(6, 6), dpi=100)
    plt.imshow(np.log1p(power), cmap='viridis')  # Use log scale for better visualization
    plt.title("FFT Power Spectrum (log scale)")
    plt.axis("off")  # Remove axes for a cleaner visualization
    
    # Save the plot to the specified directory
    plt.savefig(os.path.join(directory, "fft_power_spectrum_plot.png"))
    plt.close()

# --- Pattern Detection ---
def forms_pattern(u, v, std_threshold=0.05, fft_threshold=1e3):
    """
    Determine if a pattern is present based on std deviation or FFT energy.
    Returns True if either u or v meets pattern thresholds.
    """

    std_u = np.std(u)
    std_v = np.std(v)
    fft_u = fft_energy(u,True)[0]
    fft_v = fft_energy(v)[0]

    return (std_u > std_threshold or fft_u > fft_threshold or
            std_v > std_threshold or fft_v > fft_threshold)

# --- File and Visualization Saving ---
def save_outputs(F, k, snapshots_u, snapshots_v):
    """
    Save CSVs, images, and gif to both run-specific and centralized folders.
    """
    run_name = f"F{F:.5f}_k{k:.5f}"
    base = os.path.join("Data", run_name)
    data_dir = os.path.join(base, "Data")
    vis_dir = os.path.join(base, "Info", "visualizations")
    all_vis_dir = os.path.join("Data", "Main Info", "All Visualizations")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(all_vis_dir, exist_ok=True)

    for i, (u, v) in enumerate(zip(snapshots_u, snapshots_v)):
        np.savetxt(os.path.join(data_dir, f"u_t{i}.csv"), u, delimiter=",")
        np.savetxt(os.path.join(data_dir, f"v_t{i}.csv"), v, delimiter=",")

    # Save parameters
    with open(os.path.join(base, "Info", "parameters.csv"), "w", newline="") as f:
        csv.writer(f).writerows([["F", "k"], [F, k]])

    # Save gif
    gif_path = os.path.join(vis_dir, "evolution.gif")
    frames = []
    for u in snapshots_u:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
        ax.imshow(u, cmap='inferno')
        ax.axis('off')
        fig.tight_layout()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
        frame = frame.reshape((height, width, 4))[:, :, :3]
        frames.append(frame)
        plt.close()
    imageio.mimsave(gif_path, frames, fps=10)
    shutil.copy(gif_path, os.path.join(all_vis_dir, f"{run_name}_evolution.gif"))

    # Final frame
    final_path = os.path.join(vis_dir, "final_frame.png")
    plt.imsave(final_path, snapshots_u[-1], cmap='inferno')
    shutil.copy(final_path, os.path.join(all_vis_dir, f"{run_name}_final_frame.png"))

    # Summary plots
    mean_u = [np.mean(u) for u in snapshots_u]
    mean_v = [np.mean(v) for v in snapshots_v]
    sum_u = [np.sum(u) for u in snapshots_u]
    sum_v = [np.sum(v) for v in snapshots_v]

    plt.figure()
    plt.plot(mean_u, label="Mean U")
    plt.plot(mean_v, label="Mean V")
    plt.legend(); plt.title("Mean Evolution"); plt.xlabel("Timestep")
    plt.savefig(os.path.join(vis_dir, "mean_plot.png")); plt.close()

    plt.figure()
    plt.plot(sum_u, label="Sum U")
    plt.plot(sum_v, label="Sum V")
    plt.legend(); plt.title("Sum Evolution"); plt.xlabel("Timestep")
    plt.savefig(os.path.join(vis_dir, "sum_plot.png")); plt.close()

    graph_fft_power_spectrum(u, vis_dir)

# --- Pattern Map Plot ---
def plot_pattern_map(F_vals, k_vals, pattern_array):
    Z = np.array(pattern_array).reshape(len(F_vals), len(k_vals))
    plt.figure(figsize=(8, 6))
    plt.imshow(Z, origin='lower', cmap='Greens', extent=[k_vals[0], k_vals[-1], F_vals[0], F_vals[-1]], aspect='auto')
    plt.xlabel("k (Kill)"), plt.ylabel("F (Feed)")
    plt.title("Pattern Formation Map")
    plt.colorbar(label="Pattern = 1, No Pattern = 0")
    os.makedirs("Data/Main Info", exist_ok=True)
    plt.savefig("Data/Main Info/pattern_map.png"); plt.close()
    np.savetxt("Data/Main Info/pattern_map.csv", Z, delimiter=",")

log_entries = []
pattern_flags = []

for F in tqdm(F_vals, desc="F sweep"):
    for k in tqdm(k_vals, desc=f"k sweep for F={F}", leave=False):
        su, sv = run_simulation(F, k, params)
        pattern = forms_pattern(su[-1], sv[-1])
        log_entries.append([F, k, pattern])
        pattern_flags.append(1 if pattern else 0)
        save_outputs(F, k, su, sv)

# Save master log
with open("Data/Main Info/parameter_pattern_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["F", "k", "Pattern"])
    writer.writerows(log_entries)

with open("Data/Main Info/input_parameters.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(params.keys())
    writer.writerow(params.values())

plot_pattern_map(F_vals, k_vals, pattern_flags)
