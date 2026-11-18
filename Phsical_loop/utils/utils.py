import torch
import random

def create_circle_image(H=770, W=930, center=[770//2, 930//2], radius=100): 
    img = torch.zeros((H, W), dtype=torch.float32)
    cy, cx = center
    yy, xx = torch.meshgrid(
        torch.arange(H), 
        torch.arange(W), 
        indexing='ij' 
    )
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    mask = dist2 <= radius ** 2
    img[mask] = 1.0
    return img

def create_square_image(inputs, background, H=770, W=930, center=[770//2, 930//2], radius=100): 
    img = background
    cy, cx = center
    img[cy-radius:cy+radius, cx-radius:cx+radius] = inputs
    return img

def create_square_gt(H=770, W=930, center=[770//2, 930//2], radius=100): 
    img = torch.zeros((H, W), dtype=torch.float32)
    cy, cx = center
    img[cy-radius:cy+radius, cx-radius:cx+radius] = 1
    return img

def create_square_image_noise(inputs, H=770, W=930, center=[770//2, 930//2], radius=100): 
    img = torch.rand(H, W) * 0.7
    cy, cx = center
    img[cy-radius:cy+radius, cx-radius:cx+radius] = inputs
    return img

def print_trainable_parameters(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={tuple(param.shape)}")

import torch

def generate_grid_batch(
    objects,
    labels,
    device,
    batch_size=16,
    H=770,
    W=930,
    grid_rows=4,
    grid_cols=4,
    radius=100,
    binarize_gt=True,
):
    """
    Generate a batch of images where each image is a H x W canvas containing
    a grid of (grid_rows x grid_cols) possible cell locations. For each image,
    1-3 random cells are chosen and filled with shapes.

    - ground_truth: sum (or binary mask) of circular shapes.
    - objects_batch: sum of square shapes created from `objects[b, 0]`.

    Args:
        objects: input tensor used by `create_square_image`, expected to have
                 at least shape [batch_size, 1, ...] so that objects[b, 0] is valid.
        device: torch.device to place tensors on.
        batch_size: number of images in the batch.
        H, W: height and width of the full image.
        grid_rows, grid_cols: grid size (e.g., 4x4).
        radius: radius parameter passed to the shape-creation functions.
        binarize_gt: if True, ground_truth is converted to a {0,1} mask.

    Returns:
        ground_truth: (batch_size, H, W) tensor
        objects_batch: (batch_size, H, W) tensor, normalized per image
    """

    # Initialize empty tensors for ground truth and input objects
    ground_truth = torch.zeros(objects.shape[0], H, W, device=device)
    objects_batch = torch.rand(objects.shape[0], H, W, device=device) * 0.7
    labels_batch = torch.zeros(objects.shape[0], device=device, dtype=torch.long)

    # Precompute grid cell centers (same logic as your original code)
    # col: 0..grid_cols-1 -> x = 135 + 160 * col
    # row: 0..grid_rows-1 -> y = 115 + 230 * row
    xs = 75 + 150 * torch.arange(grid_cols, device=device)
    ys = 75 + 150 * torch.arange(grid_rows, device=device)

    for b in range(objects.shape[0]):
        # Randomly select how many objects to place in this image (1 to 3)
        num_objs = torch.randint(3, 4, (1,), device=device).item()

        # Randomly choose `num_objs` distinct cells among grid_rows * grid_cols
        num_cells = grid_rows * grid_cols
        chosen_cells = torch.randperm(num_cells, device=device)[:num_objs]
        random_chossen_label = []
        for cell in chosen_cells:
            # Compute row and column index of this cell
            row = (cell // grid_cols).item()
            col = (cell % grid_cols).item()

            # Compute center coordinates for this cell
            center = [xs[col].item(), ys[row].item()]

            # Create ground truth circle image
            gt_img = create_circle_image(
                H=H,
                W=W,
                center=center,
                radius=radius - 10
            ).to(device)

            # Create object (square) image from the given objects tensor
            object_random = torch.randint(1, objects.shape[0], (1,), device=device).item()
            random_chossen_label.append(object_random)
            objects_batch[b] = create_square_image(
                objects[object_random, 0],
                background=objects_batch[b],
                H=H,
                W=W,
                center=center,
                radius=radius
            ).to(device)

            # Add them to the current image (multiple shapes will accumulate)
            ground_truth[b] += gt_img
        label = random.choice(random_chossen_label)
        labels_batch[b] = labels[label]

    # Optionally convert ground truth to a binary mask
    if binarize_gt:
        ground_truth = (ground_truth > 0).float()

    # Normalize objects_batch per image by its maximum absolute value
    max_vals = torch.amax(torch.abs(objects_batch), dim=(-2, -1), keepdim=True)
    objects_batch = objects_batch / (max_vals + 1e-8)

    return ground_truth, objects_batch, labels_batch
