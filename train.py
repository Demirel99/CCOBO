#train.py
"""
Iterative Crowd Counting Model Training Script
"""
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR # Added for Cosine Annealing
from torch.cuda.amp import GradScaler, autocast # Added for AMP
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random

# Import from local modules
from config import (
    DEVICE, SEED, TOTAL_ITERATIONS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    VALIDATION_INTERVAL, VALIDATION_BATCHES, # SCHEDULER_PATIENCE, # Removed SCHEDULER_PATIENCE
    IMAGE_DIR_TRAIN_VAL, GT_DIR_TRAIN_VAL, OUTPUT_DIR, LOG_FILE_PATH, BEST_MODEL_PATH,
    AUGMENTATION_SIZE, MODEL_INPUT_SIZE, GT_PSF_SIGMA
)
from utils import set_seed, find_and_sort_paths, split_train_val
from dataset import generate_batch, generate_train_sample
from model import VGG19FPNASPP
from losses import kl_divergence_loss

def train():
    """Main training function."""
    print("Setting up training...")
    set_seed(SEED) # Set seed before creating model and optimizer

    # --- Data Paths ---
    sorted_image_paths_train_val = find_and_sort_paths(IMAGE_DIR_TRAIN_VAL, '*.jpg')
    sorted_gt_paths_train_val = find_and_sort_paths(GT_DIR_TRAIN_VAL, '*.mat')

    if not sorted_image_paths_train_val or not sorted_gt_paths_train_val:
        raise FileNotFoundError("Training/Validation images or ground truth files not found. Check paths in config.py.")

    # --- Train/Val Split ---
    train_image_paths, train_gt_paths, val_image_paths, val_gt_paths = split_train_val(
        sorted_image_paths_train_val, sorted_gt_paths_train_val, val_ratio=0.1, seed=SEED
    )

    if not train_image_paths or not val_image_paths:
        raise ValueError("Train or validation set is empty after splitting. Check dataset size and val_ratio.")

    # --- Model, Optimizer, Scheduler ---
    model = VGG19FPNASPP().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # OLD Scheduler:
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
    #                                                patience=SCHEDULER_PATIENCE, verbose=True)
    # NEW Scheduler: CosineAnnealingLR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TOTAL_ITERATIONS, eta_min=1e-6)


    # Initialize GradScaler for AMP, enabled only if using CUDA
    use_amp = DEVICE.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        print("Using Automatic Mixed Precision (AMP).")

    # --- Logging and Tracking ---
    best_val_loss = float('inf')
    iterations_list = []
    train_loss_list = []
    val_loss_list = []

    # Clear log file if it exists
    if os.path.exists(LOG_FILE_PATH):
        try:
            os.remove(LOG_FILE_PATH)
        except OSError as e:
            print(f"Warning: Could not remove existing log file: {e}")


    # #######################
    # # Training Loop
    # #######################
    print("Starting training...")
    pbar = tqdm(range(1, TOTAL_ITERATIONS + 1), desc=f"Iteration 1/{TOTAL_ITERATIONS}", unit="iter")

    train_loss_accum = 0.0
    samples_in_accum = 0

    for iteration in pbar:
        model.train()

        # --- Training Step ---
        img_batch, in_psf_batch, tgt_psf_batch = generate_batch(
            train_image_paths, train_gt_paths, BATCH_SIZE,
            generation_fn=generate_train_sample,
            augment_size=AUGMENTATION_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            psf_sigma=GT_PSF_SIGMA
        )

        if img_batch is None:
            print(f"Warning: Failed to generate training batch at iteration {iteration}. Skipping.")
            continue

        img_batch = img_batch.to(DEVICE)
        in_psf_batch = in_psf_batch.to(DEVICE)
        tgt_psf_batch = tgt_psf_batch.to(DEVICE)

        optimizer.zero_grad()

        # Forward pass with autocast
        with autocast(enabled=use_amp):
            predicted_psf = model(img_batch, in_psf_batch)
            loss = kl_divergence_loss(predicted_psf, tgt_psf_batch)

        # Scale loss, backward pass, and optimizer step
        scaler.scale(loss).backward()
        # Optional: If using gradient clipping, unscale gradients before clipping
        # scaler.unscale_(optimizer)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        # Step the scheduler after each optimizer update for CosineAnnealingLR
        scheduler.step()

        train_loss_accum += loss.item() * img_batch.size(0)
        samples_in_accum += img_batch.size(0)

        # --- Validation Step ---
        if iteration % VALIDATION_INTERVAL == 0:
            avg_train_loss = train_loss_accum / samples_in_accum if samples_in_accum > 0 else 0.0

            # Save RNG state
            rng_state = {
                'random': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            }

            model.eval()
            total_val_loss = 0.0
            total_val_samples = 0
            with torch.no_grad():
                for i in range(VALIDATION_BATCHES):
                    val_seed = SEED + iteration + i # Consistent seed for validation batches across runs
                    set_seed(val_seed) # Seed for batch generation

                    val_img, val_in_psf, val_tgt_psf = generate_batch(
                        val_image_paths, val_gt_paths, BATCH_SIZE,
                        generation_fn=generate_train_sample,
                        augment_size=AUGMENTATION_SIZE,
                        model_input_size=MODEL_INPUT_SIZE,
                        psf_sigma=GT_PSF_SIGMA
                    )

                    if val_img is None: continue # Skip if batch generation failed

                    val_img = val_img.to(DEVICE)
                    val_in_psf = val_in_psf.to(DEVICE)
                    val_tgt_psf = val_tgt_psf.to(DEVICE)

                    # Forward pass with autocast for validation (no scaling needed for gradients)
                    with autocast(enabled=use_amp):
                        val_pred_psf = model(val_img, val_in_psf)
                        batch_loss = kl_divergence_loss(val_pred_psf, val_tgt_psf)
                    
                    total_val_loss += batch_loss.item() * val_img.size(0)
                    total_val_samples += val_img.size(0)

            # Restore RNG state
            random.setstate(rng_state['random'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if rng_state['cuda'] and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng_state['cuda'])
            set_seed(SEED + iteration + VALIDATION_BATCHES + 1) # Reseed for next training iterations

            average_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else float('inf')

            # --- Logging & Checkpointing ---
            iterations_list.append(iteration)
            train_loss_list.append(avg_train_loss)
            val_loss_list.append(average_val_loss)

            log_message = (f"Iter [{iteration}/{TOTAL_ITERATIONS}] | "
                           f"Train Loss: {avg_train_loss:.4f} | "
                           f"Val Loss: {average_val_loss:.4f} | "
                           f"LR: {optimizer.param_groups[0]['lr']:.4e}") # This will now show the cosine annealed LR
            print(f"\n{log_message}") # Print to console (with newline)

            with open(LOG_FILE_PATH, "a") as log_file:
                log_file.write(log_message + "\n")

            # ReduceLROnPlateau scheduler step was here, no longer needed for CosineAnnealingLR
            # scheduler.step(average_val_loss) 

            if average_val_loss < best_val_loss:
                best_val_loss = average_val_loss
                torch.save(model.state_dict(), BEST_MODEL_PATH)
                print(f"    -> New best model saved with Val Loss: {best_val_loss:.4f}")

            train_loss_accum = 0.0 # Reset train loss accumulator
            samples_in_accum = 0

        # Update progress bar
        pbar.set_description(f"Iter {iteration}/{TOTAL_ITERATIONS} | Last Batch Loss: {loss.item():.4f}")

    print("Training complete!")
    pbar.close()

    # --- Plotting Training Curves ---
    print("Generating training plots...")
    plt.figure(figsize=(10, 5))
    plt.plot(iterations_list, train_loss_list, label='Train Loss')
    plt.plot(iterations_list, val_loss_list, label='Validation Loss')
    plt.title("Training and Validation Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("KL Divergence Loss")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=0) # Ensure y-axis starts at 0 for loss
    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "training_loss_plot.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Plots saved to: {OUTPUT_DIR}")
    print(f"Log file saved to: {LOG_FILE_PATH}")
    print(f"Best model saved to: {BEST_MODEL_PATH} (Val Loss: {best_val_loss:.4f})")

if __name__ == "__main__":
    train()
