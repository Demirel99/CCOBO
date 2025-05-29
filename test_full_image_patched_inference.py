# test_full_image_patched_inference.py
"""
Test script to run iterative inference on a full image by dividing it into
patches, processing each patch, and combining the results.
Clears its specific output directory before running.
Prints the total number of GT points in the original image.
"""
import os
import cv2
import torch
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import glob
import shutil

# --- Configuration ---
from config import (
    DEVICE, MODEL_INPUT_SIZE, IMAGE_DIR_TEST, GT_DIR_TEST,
    BEST_MODEL_PATH, OUTPUT_DIR, GT_PSF_SIGMA
)
from model import VGG19FPNASPP

# ImageNet Mean/Std for Normalization/Unnormalization
IMG_MEAN_CPU = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD_CPU = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# --- Helper Functions ---

def load_gt_points(gt_path):
    if not os.path.exists(gt_path): return np.array([])
    try:
        mat_data = loadmat(gt_path)
        if 'image_info' in mat_data: return mat_data['image_info'][0, 0][0, 0][0].astype(np.float32)
        if 'annPoints' in mat_data: return mat_data['annPoints'].astype(np.float32)
        for _, value in mat_data.items():
            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] == 2: return value.astype(np.float32)
        return np.array([])
    except Exception: return np.array([])

def create_input_psf_from_points(points_list, shape, sigma):
    h, w = shape
    delta_map = np.zeros((h, w), dtype=np.float32)
    if not points_list: return delta_map
    for x, y in points_list:
        x_coord, y_coord = np.clip(int(round(x)), 0, w - 1), np.clip(int(round(y)), 0, h - 1)
        delta_map[y_coord, x_coord] += 1.0
    input_psf = gaussian_filter(delta_map, sigma=sigma, order=0, mode='constant', cval=0.0)
    max_val = np.max(input_psf)
    if max_val > 1e-7: input_psf /= max_val
    return input_psf

def get_patch_coordinates(img_h, img_w, patch_size):
    """
    Generates coordinates (x_start, y_start, x_end, y_end) for patches.
    Assumes img_h, img_w are dimensions of the (potentially padded) image
    and are >= patch_size.
    """
    coords = []
    y_starts = sorted(list(set(list(range(0, img_h - patch_size, patch_size)) + ([img_h - patch_size] if img_h > 0 else []))))
    x_starts = sorted(list(set(list(range(0, img_w - patch_size, patch_size)) + ([img_w - patch_size] if img_w > 0 else []))))
    
    if not y_starts and img_h > 0 : y_starts = [0] # Handle case where img_h <= patch_size
    if not x_starts and img_w > 0 : x_starts = [0] # Handle case where img_w <= patch_size


    for sy in y_starts:
        for sx in x_starts:
            coords.append((sx, sy, sx + patch_size, sy + patch_size))
    return coords

def perform_iterative_inference_on_patch(
    model,
    image_patch_tensor_batch, # Pre-normalized, batched tensor for the patch
    gt_points_local_to_patch, # Numpy array (N,2) of GT points, coords local to patch
    num_iterations_fallback=0,
    psf_sigma=GT_PSF_SIGMA,
    model_input_size=MODEL_INPUT_SIZE,
    device=DEVICE
):
    predicted_points_local = []
    num_gt_in_patch = len(gt_points_local_to_patch)
    num_iterations = num_gt_in_patch if num_gt_in_patch > 0 else num_iterations_fallback

    if num_iterations == 0: # If fallback is 0 and no GT, predict 0 points
        return []

    for _ in range(num_iterations):
        current_input_psf_np = create_input_psf_from_points(
            predicted_points_local,
            shape=(model_input_size, model_input_size),
            sigma=psf_sigma
        )
        current_input_psf_tensor = torch.from_numpy(current_input_psf_np).float().unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            predicted_output_psf_tensor = model(image_patch_tensor_batch, current_input_psf_tensor)

        output_psf_np = predicted_output_psf_tensor.squeeze().cpu().numpy()
        max_yx = np.unravel_index(np.argmax(output_psf_np), output_psf_np.shape)
        pred_y, pred_x = max_yx[0], max_yx[1]
        predicted_points_local.append((pred_x, pred_y))
    return predicted_points_local

# --- Main Function ---
def main():
    print("--- Full Image Patched Iterative Inference Script ---")
    patch_level_visuals_enabled = False # Set to True to save individual patch summaries
    num_pred_fallback_per_patch = 3     # Number of points to predict if a patch has no GT

    # Output Directory Setup
    script_output_dir = os.path.join(OUTPUT_DIR, "full_image_patched_outputs")
    if os.path.exists(script_output_dir):
        print(f"Clearing existing output directory: {script_output_dir}")
        try: shutil.rmtree(script_output_dir)
        except OSError as e:
            print(f"Error clearing directory: {e}. Please close files and retry.")
            return
    os.makedirs(script_output_dir, exist_ok=True)
    print(f"Outputs will be saved in: {script_output_dir}")

    # Load Model
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"Error: Model not found at {BEST_MODEL_PATH}"); return
    model = VGG19FPNASPP().to(DEVICE)
    model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model loaded.")

    # Select Image
    test_image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR_TEST, '*.jpg')))
    if not test_image_paths: print(f"No test images in {IMAGE_DIR_TEST}"); return
    image_path = test_image_paths[25] # Process the first image
    img_filename = os.path.basename(image_path)
    img_filestem = os.path.splitext(img_filename)[0]
    gt_path = os.path.join(GT_DIR_TEST, "GT_" + img_filestem + ".mat")
    print(f"Processing image: {img_filename}")

    # Load Original Image and GT
    img_orig_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    gt_orig_global = load_gt_points(gt_path) # (N,2) array (x,y)
    
    # Print the number of GT points
    num_gt_points_original_image = len(gt_orig_global)
    print(f"Total number of GT points in the original image: {num_gt_points_original_image}")


    # Pad original image if smaller than patch size
    h_orig, w_orig = img_orig_rgb.shape[:2]
    pad_h_bottom = max(0, MODEL_INPUT_SIZE - h_orig)
    pad_w_right = max(0, MODEL_INPUT_SIZE - w_orig)
    
    img_to_process = cv2.copyMakeBorder(img_orig_rgb, 0, pad_h_bottom, 0, pad_w_right,
                                        cv2.BORDER_CONSTANT, value=[0,0,0]) # Pad with black
    h_proc, w_proc = img_to_process.shape[:2]

    # Get Patch Coordinates for the (padded) image to process
    patch_coords_list = get_patch_coordinates(h_proc, w_proc, MODEL_INPUT_SIZE)
    print(f"Generated {len(patch_coords_list)} patches for the image.")

    all_predicted_points_global = [] # Store all (x,y) predictions in original image coords

    for i, (x_start, y_start, x_end, y_end) in enumerate(patch_coords_list):
        print(f"  Processing patch {i+1}/{len(patch_coords_list)}: region ({x_start},{y_start}) to ({x_end},{y_end})")

        # Extract patch image data
        img_patch_np = img_to_process[y_start:y_end, x_start:x_end]

        # Normalize patch and convert to tensor
        patch_tensor_np = img_patch_np.astype(np.float32) / 255.0
        patch_tensor_chw = torch.from_numpy(patch_tensor_np).permute(2, 0, 1)
        patch_tensor_norm = (patch_tensor_chw - IMG_MEAN_CPU) / IMG_STD_CPU
        patch_tensor_batch = patch_tensor_norm.unsqueeze(0).to(DEVICE)

        # Get GT points local to this patch
        gt_local_to_patch = []
        if gt_orig_global.size > 0:
            for gx, gy in gt_orig_global:
                if x_start <= gx < x_end and y_start <= gy < y_end:
                    lx, ly = gx - x_start, gy - y_start
                    gt_local_to_patch.append((lx, ly))
        gt_local_to_patch = np.array(gt_local_to_patch)

        # Perform iterative inference on the patch
        predicted_points_local = perform_iterative_inference_on_patch(
            model, patch_tensor_batch, gt_local_to_patch,
            num_iterations_fallback=num_pred_fallback_per_patch,
            psf_sigma=GT_PSF_SIGMA, model_input_size=MODEL_INPUT_SIZE, device=DEVICE
        )

        # Convert local predictions to global coordinates and store
        for plx, ply in predicted_points_local:
            pgx, pgy = plx + x_start, ply + y_start
            if pgx < w_orig and pgy < h_orig:
                 all_predicted_points_global.append((pgx, pgy))

        if patch_level_visuals_enabled and predicted_points_local:
            plt.figure(figsize=(6,6))
            display_patch_tensor = patch_tensor_batch.squeeze(0).cpu() * IMG_STD_CPU + IMG_MEAN_CPU
            display_patch_np = np.clip(display_patch_tensor.permute(1, 2, 0).numpy(), 0, 1)
            plt.imshow(display_patch_np)
            if gt_local_to_patch.size > 0:
                plt.scatter(gt_local_to_patch[:,0], gt_local_to_patch[:,1], s=30, facecolors='none', edgecolors='lime', lw=1, label='Local GT')
            preds_local_np = np.array(predicted_points_local)
            plt.scatter(preds_local_np[:,0], preds_local_np[:,1], s=20, c='red', marker='x', label='Local Preds')
            plt.title(f"Patch {i+1} ({x_start},{y_start}) Results")
            plt.axis('off'); plt.legend()
            patch_plot_path = os.path.join(script_output_dir, f"{img_filestem}_patch_{i+1:03d}.png")
            plt.savefig(patch_plot_path); plt.close()

    # Final Visualization
    plt.figure(figsize=(12, 12 * h_orig / w_orig if w_orig > 0 else 12))
    plt.imshow(img_orig_rgb)
    if gt_orig_global.size > 0:
        plt.scatter(gt_orig_global[:, 0], gt_orig_global[:, 1], s=30, facecolors='none', edgecolors='lime', linewidths=1.5, label=f'Ground Truth ({num_gt_points_original_image})')
    if all_predicted_points_global:
        preds_global_np = np.array(all_predicted_points_global)
        plt.scatter(preds_global_np[:, 0], preds_global_np[:, 1], s=20, c='red', marker='x', label=f'Predicted ({len(preds_global_np)})')
    plt.title(f"Full Image Predictions for {img_filename}")
    plt.axis('off'); plt.legend()
    final_summary_path = os.path.join(script_output_dir, f"{img_filestem}_full_summary.png")
    plt.savefig(final_summary_path); plt.close()
    print(f"\nFull image processing complete. Total points predicted: {len(all_predicted_points_global)}")
    print(f"Final summary plot saved to: {final_summary_path}")

if __name__ == "__main__":
    main()
