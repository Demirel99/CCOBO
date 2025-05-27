# dataset.py
"""
Dataset preparation, augmentation, and sample generation functions.
"""
import numpy as np
import cv2
import random
import torch
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
import os # Added for test code
import glob # Added for test code
import matplotlib.pyplot as plt # Added for test code

# Import from config
from config import (AUGMENTATION_SIZE, MODEL_INPUT_SIZE, MIN_DIM_RESCALE, GT_PSF_SIGMA, IMAGE_DIR_TRAIN_VAL, GT_DIR_TRAIN_VAL, DEVICE)


# --- ImageNet Mean/Std for Normalization ---
IMG_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMG_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def prepare_data_augmentations(image, gt_coor, target_size=AUGMENTATION_SIZE, min_dim=MIN_DIM_RESCALE):
    """
    Applies random scaling, cropping, and flipping augmentations.
    Outputs an image and corresponding coordinates at the target_size.

    Args:
        image (np.ndarray): Input image (H, W, C).
        gt_coor (np.ndarray): Ground truth coordinates (N, 2) as (x, y).
        target_size (int): The size of the square crop to extract.
        min_dim (int): Minimum dimension allowed after scaling.

    Returns:
        tuple: (augmented_image, augmented_gt_coor)
               augmented_image is (target_size, target_size, C)
               augmented_gt_coor is (M, 2) for points within the final crop.
    """
    if image is None:
        print("Warning: prepare_data_augmentations received None image.")
        return None, None
    h, w, c = image.shape
    if c != 3:
        print(f"Warning: Expected a 3-channel image (H, W, 3), but got {c} channels. Attempting to proceed.")
        # Potentially handle grayscale or other formats if needed, or raise error
        # For now, let's just print a warning.

    # 1. Random Scale
    scale_factor = random.uniform(0.7, 1.3)
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # Enforce minimum dimension
    # Check min(h, w) > 0 before division
    if min(h, w) > 0 and min(new_h, new_w) < min_dim:
        scale_factor = min_dim / min(h, w)
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    elif min(h, w) == 0:
        print("Warning: Image has zero dimension before scaling.")
        return None, None # Handle zero dimension images

    # Ensure new dimensions are positive
    new_h = max(1, new_h)
    new_w = max(1, new_w)

    try:
        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"Error during cv2.resize: {e}. Original shape: {(h,w)}, Target shape: {(new_h, new_w)}")
        return None, None

    scaled_gt_coor = gt_coor * scale_factor if gt_coor is not None and gt_coor.size > 0 else np.array([]) # Handle empty/None GT

    # 2. Random Crop (to target_size)
    curr_h, curr_w = scaled_image.shape[:2]
    crop_h = min(target_size, curr_h)
    crop_w = min(target_size, curr_w)

    cropped_gt_coor = np.array([]) # Initialize

    if curr_h >= target_size and curr_w >= target_size:
        x_min = random.randint(0, curr_w - target_size)
        y_min = random.randint(0, curr_h - target_size)
        cropped_image = scaled_image[y_min : y_min + target_size, x_min : x_min + target_size]
        # Adjust coordinates
        if scaled_gt_coor.size > 0:
            keep_mask = (scaled_gt_coor[:, 0] >= x_min) & (scaled_gt_coor[:, 0] < x_min + target_size) & \
                        (scaled_gt_coor[:, 1] >= y_min) & (scaled_gt_coor[:, 1] < y_min + target_size)
            cropped_gt_coor = scaled_gt_coor[keep_mask]
            if cropped_gt_coor.size > 0: # Check if any points remain
                cropped_gt_coor[:, 0] -= x_min
                cropped_gt_coor[:, 1] -= y_min
        # else: cropped_gt_coor remains empty
        final_h, final_w = target_size, target_size
    else:
        # If image is smaller than target_size after scaling, resize up (less ideal than padding)
        print(f"Warning: Scaled image ({curr_h}x{curr_w}) smaller than target ({target_size}x{target_size}). Resizing up.")
        try:
             cropped_image = cv2.resize(scaled_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        except cv2.error as e:
             print(f"Error during cv2.resize (upscaling): {e}. Scaled shape: {(curr_h, curr_w)}, Target shape: {(target_size, target_size)}")
             return None, None

        if curr_w > 0 and curr_h > 0 and scaled_gt_coor.size > 0:
            scale_x_final = target_size / float(curr_w)
            scale_y_final = target_size / float(curr_h)
            cropped_gt_coor = scaled_gt_coor.copy()
            cropped_gt_coor[:, 0] *= scale_x_final
            cropped_gt_coor[:, 1] *= scale_y_final
            # Clip coordinates
            cropped_gt_coor[:, 0] = np.clip(cropped_gt_coor[:, 0], 0, target_size - 1)
            cropped_gt_coor[:, 1] = np.clip(cropped_gt_coor[:, 1], 0, target_size - 1)
        # else: cropped_gt_coor remains empty
        final_h, final_w = target_size, target_size

    # 3. Random Horizontal Flip
    if random.random() < 0.5:
        cropped_image = cv2.flip(cropped_image, 1) # 1 = horizontal flip
        if cropped_gt_coor.size > 0:
            cropped_gt_coor[:, 0] = (final_w - 1) - cropped_gt_coor[:, 0]

    # Ensure output shape matches target size
    if cropped_image.shape[0] != target_size or cropped_image.shape[1] != target_size:
         print(f"Warning: Final augmented image shape {cropped_image.shape} does not match target {target_size}x{target_size}. Resizing again.")
         try:
            cropped_image = cv2.resize(cropped_image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
         except cv2.error as e:
            print(f"Error during final cv2.resize: {e}. Shape before: {cropped_image.shape}, Target shape: {(target_size, target_size)}")
            return None, None


    return cropped_image, cropped_gt_coor


def generate_single_psf(coord, image_shape, sigma):
    """Generates a single normalized Gaussian PSF centered at coord."""
    height, width = image_shape
    psf = np.zeros((height, width), dtype=np.float32)
    # Ensure coordinates are within bounds
    x = np.clip(int(round(coord[0])), 0, width - 1)
    y = np.clip(int(round(coord[1])), 0, height - 1)
    psf[y, x] = 1.0
    # Apply Gaussian filter
    # Note: sigma here relates to the filter size, not std deviation directly in pixels
    # A sigma of ~1 roughly corresponds to a 3x3 kernel effect, sigma=3 is wider
    psf = gaussian_filter(psf, sigma=sigma, order=0, mode='constant', cval=0.0)
    psf_sum = np.sum(psf)
    if psf_sum > 1e-7: # Avoid division by zero or near-zero
        psf /= psf_sum
    return psf

def get_center_crop_coords(image_size, crop_size):
    """Calculates the top-left (y, x) coordinates for a center crop."""
    img_h, img_w = image_size
    crop_h, crop_w = crop_size
    start_y = max(0, (img_h - crop_h) // 2) # Ensure non-negative start
    start_x = max(0, (img_w - crop_w) // 2)
    return start_y, start_x

def check_any_coord_in_center(coordinates, image_shape, center_crop_shape):
    """Checks if at least one coordinate falls within the central crop area."""
    if coordinates is None or coordinates.shape[0] == 0: return False
    img_h, img_w = image_shape
    crop_h, crop_w = center_crop_shape
    start_y, start_x = get_center_crop_coords(image_shape, center_crop_shape)
    # Ensure end coords do not exceed image bounds implicitly handled by check
    end_y, end_x = start_y + crop_h, start_x + crop_w

    for x, y in coordinates:
        # Check if point (x, y) is within the rectangle [start_x, end_x) and [start_y, end_y)
        if start_x <= x < end_x and start_y <= y < end_y:
            return True
    return False


def get_random_coord_index_in_center(coordinates, image_shape, center_crop_shape):
    """Gets the index of a random coordinate falling within the central crop."""
    if coordinates is None or coordinates.shape[0] == 0: return None
    img_h, img_w = image_shape
    crop_h, crop_w = center_crop_shape
    start_y, start_x = get_center_crop_coords(image_shape, center_crop_shape)
    end_y, end_x = start_y + crop_h, start_x + crop_w

    indices_in_center = [
        i for i, (x, y) in enumerate(coordinates)
        if start_x <= x < end_x and start_y <= y < end_y
    ]

    return random.choice(indices_in_center) if indices_in_center else None


def generate_train_sample(image_paths, gt_paths, augment_size=AUGMENTATION_SIZE, model_input_size=MODEL_INPUT_SIZE, psf_sigma=GT_PSF_SIGMA):
    """
    Generates a single training sample (image, input_psf, target_psf).
    Processes points starting from bottom-left, going right.
    """
    max_retries = 10 # Prevent infinite loops on problematic data
    for _ in range(max_retries):
        # 1. Load Image and GT
        rand_idx = random.randint(0, len(image_paths) - 1)
        image_path = image_paths[rand_idx]
        # Construct expected GT path (assuming name match except extension)
        img_filename = os.path.basename(image_path)
        gt_filename = "GT_" + os.path.splitext(img_filename)[0] + ".mat"
        gt_path = os.path.join(os.path.dirname(gt_paths[0]), gt_filename) # Use dirname from gt_paths list

        if not os.path.exists(gt_path):
             # Fallback using index if naming convention fails (less robust)
             if rand_idx < len(gt_paths):
                 gt_path_fallback = gt_paths[rand_idx]
                 if os.path.exists(gt_path_fallback):
                     gt_path = gt_path_fallback
                 else:
                     print(f"Warning: GT file not found for {image_path} using convention ({gt_filename}) or index. Skipping.")
                     continue
             else:
                 print(f"Warning: GT index out of bounds for {image_path}. Skipping.")
                 continue


        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {image_path}. Skipping.")
            continue
        # Ensure image is RGB
        if len(image.shape) == 2: # Grayscale
            print(f"Warning: Image {image_path} is grayscale. Converting to RGB.")
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4: # RGBA
             print(f"Warning: Image {image_path} has 4 channels (RGBA?). Converting to RGB.")
             image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] != 3:
             print(f"Warning: Image {image_path} has unexpected shape {image.shape}. Skipping.")
             continue
        # Standard BGR to RGB conversion
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        try:
            mat_data = loadmat(gt_path)
            # Navigate the typical structure for ShanghaiTech
            # Check if 'image_info' exists, otherwise look for common alternatives like 'annPoints'
            if 'image_info' in mat_data:
                 gt_coor = mat_data['image_info'][0, 0][0, 0][0].astype(np.float32) # Nx2 (x, y)
            elif 'annPoints' in mat_data:
                 gt_coor = mat_data['annPoints'].astype(np.float32)
            else:
                 # Try to find any array that looks like coordinates (Nx2)
                 found_coords = False
                 for key, value in mat_data.items():
                     if isinstance(value, np.ndarray) and len(value.shape) == 2 and value.shape[1] == 2:
                         gt_coor = value.astype(np.float32)
                         print(f"Warning: Found coordinates in unexpected key '{key}' for {gt_path}.")
                         found_coords = True
                         break
                 if not found_coords:
                     print(f"Warning: Could not find coordinate data in expected keys ('image_info', 'annPoints') or plausible alternatives in {gt_path}. Skipping.")
                     continue

        except (FileNotFoundError) as e:
             print(f"Warning: File not found {gt_path}: {e}. Skipping.")
             continue
        except (KeyError, IndexError, TypeError) as e:
             print(f"Warning: Error parsing .mat structure {gt_path}: {e}. Skipping.")
             continue
        except Exception as e: # Catch other potential load errors
             print(f"Warning: Unexpected error loading/parsing .mat {gt_path}: {e}. Skipping.")
             continue


        if gt_coor.shape[0] == 0:
            # print(f"Debug: Image {image_path} has 0 points. Retrying sample generation.")
            continue # Skip images with no points, try another image

        # 2. Apply Augmentations
        aug_image, aug_gt_coor = prepare_data_augmentations(image, gt_coor.copy(), target_size=augment_size)

        if aug_image is None or aug_gt_coor is None:
            # print(f"Debug: Augmentation failed for {image_path}. Retrying sample generation.")
            continue # Augmentation failed

        # Check if any coordinates remain after augmentation
        if aug_gt_coor.shape[0] < 1:
            # print(f"Debug: No points remaining after augmentation for {image_path}. Retrying sample generation.")
            continue # Need at least one point for target

        img_h, img_w = aug_image.shape[:2]
        if img_h == 0 or img_w == 0:
             print(f"Warning: Augmented image has zero dimension for {image_path}. Skipping.")
             continue


        # 3. Check for points in center crop
        center_crop_shape = (model_input_size, model_input_size)
        if not check_any_coord_in_center(aug_gt_coor, (img_h, img_w), center_crop_shape):
            # print(f"Debug: No points in center crop for {image_path}. Retrying sample generation.")
            continue # Target point must be predictable within the final crop

        # 4. Select Target Point (randomly from those in the center)
        target_coord_idx = get_random_coord_index_in_center(aug_gt_coor, (img_h, img_w), center_crop_shape)
        if target_coord_idx is None:
             # This should theoretically not happen if check_any_coord_in_center passed, but check anyway
             # print(f"Debug: Failed to get target index in center for {image_path}. Retrying sample generation.")
             continue

        # 5. Prepare Input and Target PSFs based on NEW sorting order
        # --- MODIFICATION START ---
        # Sort points: primary key = y descending (bottom first), secondary key = x ascending (left first)
        # Achieved by sorting ascending on (-y, x) using np.lexsort
        # Note: np.lexsort uses the *last* key as primary.
        sorted_indices = np.lexsort((aug_gt_coor[:, 0], -aug_gt_coor[:, 1])) # Sort by x (secondary), then -y (primary)
        # --- MODIFICATION END ---
        sorted_aug_gt_coor = aug_gt_coor[sorted_indices]

        # Find the index of our chosen target point *within the sorted list*
        target_coord = aug_gt_coor[target_coord_idx]
        sorted_target_idx = -1
        # Use np.isclose for floating point comparison robustness
        for i, coord in enumerate(sorted_aug_gt_coor):
             # Check if coord and target_coord are close enough
             if np.allclose(coord, target_coord, atol=1e-4): # Allow small tolerance
                 sorted_target_idx = i
                 break
        if sorted_target_idx == -1:
            print(f"Warning: Could not find the selected target coordinate {target_coord} in the sorted list for {image_path}. This is unexpected. Skipping.")
            # print("Sorted list sample:", sorted_aug_gt_coor[:5]) # Debug print
            continue

        timestep = sorted_target_idx # This is the index (0 to N-1) of the target point in the *newly sorted* list
        target_psf_full = generate_single_psf(sorted_aug_gt_coor[timestep], (img_h, img_w), psf_sigma)

        input_psf_full = np.zeros((img_h, img_w), dtype=np.float32)
        if timestep > 0:
            # Sum the PSFs of *all points before* the target point in the sorted list
            for i in range(timestep):
                input_psf_full += generate_single_psf(sorted_aug_gt_coor[i], (img_h, img_w), psf_sigma)
        # If timestep is 0, input_psf_full remains all zeros, which is correct.

        # 6. Center Crop Image and PSFs
        start_y, start_x = get_center_crop_coords((img_h, img_w), center_crop_shape)
        end_y, end_x = start_y + model_input_size, start_x + model_input_size

        # Ensure crop boundaries are valid
        if start_y < 0 or start_x < 0 or end_y > img_h or end_x > img_w:
            print(f"Warning: Invalid crop coordinates for {image_path}. Aug H/W: ({img_h},{img_w}), Crop Size: {center_crop_shape}, Start Y/X: ({start_y},{start_x}). Skipping.")
            continue

        final_image = aug_image[start_y:end_y, start_x:end_x]
        final_input_psf = input_psf_full[start_y:end_y, start_x:end_x]
        final_target_psf = target_psf_full[start_y:end_y, start_x:end_x]

        # Verify cropped shapes
        if final_image.shape[:2] != center_crop_shape or \
           final_input_psf.shape != center_crop_shape or \
           final_target_psf.shape != center_crop_shape:
            print(f"Warning: Cropped shape mismatch for {image_path}.")
            print(f"  Expected: {center_crop_shape}")
            print(f"  Got Img: {final_image.shape[:2]}, InPSF: {final_input_psf.shape}, TgtPSF: {final_target_psf.shape}")
            # Attempt to resize if slightly off? Or just skip. Skipping is safer.
            continue


        # 7. Normalize Image and PSFs
        # Convert image to float, normalize to [0, 1], then apply ImageNet normalization
        final_image_tensor = torch.from_numpy(final_image.copy()).permute(2, 0, 1).float() / 255.0
        final_image_tensor = (final_image_tensor - IMG_MEAN) / IMG_STD

        # Normalize Input PSF: Max normalization (as done in some papers)
        # Ensures the input mask values are roughly within [0, 1]
        max_val = np.max(final_input_psf)
        if max_val > 1e-7: # Avoid division by zero/small numbers
            final_input_psf = final_input_psf / max_val
        final_input_psf_tensor = torch.from_numpy(final_input_psf).float().unsqueeze(0) # Add channel dim

        # Normalize Target PSF: Ensure it sums to 1 after cropping
        # generate_single_psf already normalized it to sum=1 over the *full* augmented image.
        # Cropping might slightly change the sum. Renormalize the *cropped* version.
        target_psf_sum = np.sum(final_target_psf)
        if target_psf_sum > 1e-7: # Avoid division by zero
            final_target_psf = final_target_psf / target_psf_sum
        else:
            # If sum is near zero after cropping (e.g., Gaussian tail was tiny in crop)
            # Set to uniform zero? Or maybe this sample should be skipped earlier?
            # For now, it will be near zero, which might be okay.
             pass
        final_target_psf_tensor = torch.from_numpy(final_target_psf).float().unsqueeze(0) # Add channel dim

        # Final Safety check for shapes
        expected_shape = (model_input_size, model_input_size)
        if final_image_tensor.shape != (3, *expected_shape) or \
           final_input_psf_tensor.shape != (1, *expected_shape) or \
           final_target_psf_tensor.shape != (1, *expected_shape):
            print(f"Warning: Final shape mismatch before returning for {image_path}. Retrying.")
            print(f"  Shapes: Img={final_image_tensor.shape}, InPSF={final_input_psf_tensor.shape}, TgtPSF={final_target_psf_tensor.shape}")
            print(f"  Expected: Img=(3, {model_input_size}, {model_input_size}), PSFs=(1, {model_input_size}, {model_input_size})")
            continue

        # If all checks passed, return the sample
        return final_image_tensor, final_input_psf_tensor, final_target_psf_tensor

    # If loop finished without returning, all retries failed
    print(f"Warning: Failed to generate a valid sample after {max_retries} retries. Returning None.")
    return None, None, None


def generate_batch(image_paths, gt_paths, batch_size, generation_fn=generate_train_sample, **kwargs):
    """Generates a batch of data using the specified generation function."""
    image_batch, input_psf_batch, output_psf_batch = [], [], []
    attempts = 0
    max_attempts = batch_size * 10 # Increase max attempts slightly

    while len(image_batch) < batch_size and attempts < max_attempts:
        attempts += 1
        try:
            # Pass additional arguments like augment_size, model_input_size etc.
            sample = generation_fn(image_paths, gt_paths, **kwargs)

            if sample is not None and sample[0] is not None: # Check if sample and first element are valid
                img, in_psf, out_psf = sample
                # Basic validation of returned tensors
                if isinstance(img, torch.Tensor) and isinstance(in_psf, torch.Tensor) and isinstance(out_psf, torch.Tensor):
                    image_batch.append(img)
                    input_psf_batch.append(in_psf)
                    output_psf_batch.append(out_psf)
                else:
                    print(f"Warning: generation_fn returned non-Tensor data. Skipping sample. Types: {type(img)}, {type(in_psf)}, {type(out_psf)}")

            # Else: generation_fn returned None (e.g., bad image, no points), continue loop
        except Exception as e:
            # Log the exception traceback for debugging
            import traceback
            print(f"Error during sample generation: {e}. Skipping sample.")
            print(traceback.format_exc()) # Print stack trace
            continue # Try generating the next sample for the batch

    if not image_batch: # Handle case where batch generation failed completely
        print(f"Warning: Failed to generate any valid samples for a batch after {max_attempts} attempts.")
        return None, None, None

    # Stack the collected tensors
    try:
        final_image_batch = torch.stack(image_batch)
        final_input_psf_batch = torch.stack(input_psf_batch)
        final_output_psf_batch = torch.stack(output_psf_batch)
        return final_image_batch, final_input_psf_batch, final_output_psf_batch
    except Exception as e:
        print(f"Error during torch.stack: {e}")
        print(f"Number of samples collected: {len(image_batch)}")
        # Print shapes of individual tensors if possible
        if image_batch:
            print("Individual image shapes:", [t.shape for t in image_batch])
        if input_psf_batch:
            print("Individual input PSF shapes:", [t.shape for t in input_psf_batch])
        if output_psf_batch:
            print("Individual target PSF shapes:", [t.shape for t in output_psf_batch])
        return None, None, None


# ==============================================================================
# Test Code - Visualize a generated sample
# ==============================================================================
if __name__ == "__main__":
    print("Running dataset.py test...")

    # --- Configuration for Test ---
    # Use paths from config if available, otherwise use fallbacks defined above
    test_image_dir = IMAGE_DIR_TRAIN_VAL
    test_gt_dir = GT_DIR_TRAIN_VAL
    num_samples_to_show = 3 # How many examples to generate and show

    print(f"Using Image Dir: {test_image_dir}")
    print(f"Using GT Dir: {test_gt_dir}")
    print(f"Model Input Size: {MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE}")
    print(f"Augmentation Size: {AUGMENTATION_SIZE}x{AUGMENTATION_SIZE}")
    print(f"GT PSF Sigma: {GT_PSF_SIGMA}")

    # --- Load File Paths ---
    try:
        image_paths = sorted(glob.glob(os.path.join(test_image_dir, '*.jpg')))
        # Assume corresponding .mat files exist for GT, build list for generate_train_sample
        # generate_train_sample internally constructs the GT path based on image path
        # We just need a list of the same length for the function signature, content doesn't matter much here
        # Alternatively, explicitly list the gt files if naming convention differs wildly
        gt_paths = sorted(glob.glob(os.path.join(test_gt_dir, '*.mat'))) # Pass the actual list

        if not image_paths:
            print(f"Error: No images found in {test_image_dir}")
            exit()
        # if not gt_paths:
        #     print(f"Warning: No ground truth .mat files found directly in {test_gt_dir}. "
        #           f"generate_train_sample will try to construct paths based on image names.")
        # if len(image_paths) != len(gt_paths):
        #      print(f"Warning: Number of images ({len(image_paths)}) does not match number of GT files ({len(gt_paths)}) found.")
        #      # Proceeding anyway as generate_train_sample matches them by name.

    except Exception as e:
        print(f"Error finding dataset files: {e}")
        exit()

    print(f"Found {len(image_paths)} images.")

    # --- Generate and Visualize Samples ---
    for i in range(num_samples_to_show):
        print(f"\n--- Generating Sample {i+1}/{num_samples_to_show} ---")
        sample_data = generate_train_sample(
            image_paths,
            gt_paths, # Pass the list of GT paths
            augment_size=AUGMENTATION_SIZE,
            model_input_size=MODEL_INPUT_SIZE,
            psf_sigma=GT_PSF_SIGMA
        )

        if sample_data is None or sample_data[0] is None:
            print("Failed to generate a valid sample. Skipping visualization for this attempt.")
            continue

        img_tensor, input_psf_tensor, target_psf_tensor = sample_data

        print(f"Generated Tensor Shapes:")
        print(f"  Image:      {img_tensor.shape}")
        print(f"  Input PSF:  {input_psf_tensor.shape}")
        print(f"  Target PSF: {target_psf_tensor.shape}")

        # --- Convert Tensors for Visualization ---
        # Denormalize image
        img_vis = img_tensor.cpu() * IMG_STD.cpu() + IMG_MEAN.cpu()
        img_vis = torch.clamp(img_vis, 0, 1) # Clamp values to [0, 1]
        img_vis_np = img_vis.permute(1, 2, 0).numpy() # HWC for matplotlib

        # PSFs to numpy (remove batch and channel dims)
        input_psf_np = input_psf_tensor.squeeze().cpu().numpy()
        target_psf_np = target_psf_tensor.squeeze().cpu().numpy()

        # --- Plotting ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Generated Sample {i+1} (Bottom-Left First Sort)', fontsize=16) # Updated title

        # Image
        axes[0].imshow(img_vis_np)
        axes[0].set_title(f'Input Image ({MODEL_INPUT_SIZE}x{MODEL_INPUT_SIZE})')
        axes[0].axis('off')

        # Input PSF
        im_in = axes[1].imshow(input_psf_np, cmap='viridis', vmin=0) # Use 'viridis' or 'jet'
        axes[1].set_title(f'Input PSF (Sum of previous points)\nMax val: {np.max(input_psf_np):.4f}')
        axes[1].axis('off')
        fig.colorbar(im_in, ax=axes[1], fraction=0.046, pad=0.04)

        # Target PSF
        im_tgt = axes[2].imshow(target_psf_np, cmap='viridis', vmin=0)
        axes[2].set_title(f'Target PSF (Current point)\nSum: {np.sum(target_psf_np):.4f}')
        axes[2].axis('off')
        fig.colorbar(im_tgt, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()

    print("\nDataset test finished.")