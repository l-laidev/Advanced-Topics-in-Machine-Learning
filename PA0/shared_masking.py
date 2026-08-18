import torch


def randomly_mask(images, patch_size=16, mask_ratio=0.75, mask_value=0.0):
    """
    Randomly masks patches of an RGB image tensor.
    
    Args:
        images (torch.Tensor): Input tensor of shape (B, 3, H, W)
        patch_size (int): Height and width of the square patches
        mask_ratio (float): Fraction of patches to mask (e.g., 0.75 for 75%)
        mask_value (float): Pixel value to fill the masked patches with
        
    Returns:
        torch.Tensor: Image tensor with randomly masked patches, shape (B, 3, H, W)
    """
    B, C, H, W = images.shape
    assert H % patch_size == 0 and W % patch_size == 0, f"Image dimensions must be divisible by patch_size: {H}x{W} % {patch_size}"
    
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w
    num_masked_patches = int(mask_ratio * num_patches)
    
    # 1. Reshape into patches: (B, 3, num_patches_h, patch_size, num_patches_w, patch_size)
    patches = images.view(B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    
    # Permute to isolate the patches: (B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    
    # Flatten spatial patch grid to a sequence: (B, num_patches, C, patch_size, patch_size)
    patches = patches.reshape(B, num_patches, C, patch_size, patch_size)
    
    # 2. Generate random masking indices for each batch item independently
    # Create uniform random noise for every patch position
    noise = torch.rand(B, num_patches, device=images.device) 
    
    # Sort noise to easily pick the smallest or largest elements
    ids_shuffle = torch.argsort(noise, dim=1) 
    
    # 3. Create a binary mask: 1 for keep, 0 for remove
    mask = torch.ones(B, num_patches, device=images.device)
    
    # Set the first 'num_masked_patches' indices in the shuffled sequence to 0
    # We use scatter_ to dynamically assign zeros based on our random index sequence
    mask.scatter_(1, ids_shuffle[:, :num_masked_patches], 0.0)
    
    # 4. Reshape mask to broadcast over the color channels and patch pixels
    # (B, num_patches, 1, 1, 1)
    mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    
    # Apply mask and replace zero positions with your designated mask_value
    masked_patches = patches * mask
    if mask_value != 0.0:
        masked_patches = torch.where(mask == 1.0, masked_patches, torch.tensor(mask_value, device=images.device))
        
    # 5. Reconstruct back to original format (B, 3, H, W)
    masked_patches = masked_patches.view(B, num_patches_h, num_patches_w, C, patch_size, patch_size)
    masked_patches = masked_patches.permute(0, 3, 1, 4, 2, 5) # (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
    masked_images = masked_patches.reshape(B, C, H, W)
    
    return masked_images

def structurally_mask(images, patch_size=16, mask_ratio=0.75, mask_value=0.0):
    """
    Structurally masks a contiguous region of patches in an RGB image tensor.

    Args:
        images (torch.Tensor): Input tensor of shape (B, 3, H, W)
        patch_size (int): Height and width of square patches.
        mask_ratio (float): Fraction of patches to mask.
        mask_value (float): Pixel value used for masked patches.

    Returns:
        torch.Tensor: Structurally masked images, shape (B, 3, H, W)
    """

    B, C, H, W = images.shape

    assert H % patch_size == 0 and W % patch_size == 0, (
        f"Image dimensions must be divisible by patch_size: "
        f"{H}x{W} % {patch_size}"
    )

    num_patches_h = H // patch_size
    num_patches_w = W // patch_size
    num_patches = num_patches_h * num_patches_w

    num_masked_patches = int(mask_ratio * num_patches)

    # ---------------------------------------------------------
    # 1. Convert image into patches
    # ---------------------------------------------------------

    patches = images.view(
        B,
        C,
        num_patches_h,
        patch_size,
        num_patches_w,
        patch_size
    )

    patches = patches.permute(
        0, 2, 4, 1, 3, 5
    )

    # (B, num_patches, C, patch_size, patch_size)
    patches = patches.reshape(
        B,
        num_patches,
        C,
        patch_size,
        patch_size
    )

    # ---------------------------------------------------------
    # 2. Create STRUCTURED mask
    # ---------------------------------------------------------

    mask = torch.ones(
        B,
        num_patches_h,
        num_patches_w,
        device=images.device
    )

    # Determine approximately square region to mask
    mask_h = max(1, int(num_patches_h * (mask_ratio ** 0.5)))
    mask_w = max(1, int(num_patches_w * (mask_ratio ** 0.5)))

    # Random starting position for the structural block
    for b in range(B):

        start_h = torch.randint(
            0,
            num_patches_h - mask_h + 1,
            (1,),
            device=images.device
        ).item()

        start_w = torch.randint(
            0,
            num_patches_w - mask_w + 1,
            (1,),
            device=images.device
        ).item()

        # Mask a contiguous rectangular region
        mask[
            b,
            start_h:start_h + mask_h,
            start_w:start_w + mask_w
        ] = 0.0

    # ---------------------------------------------------------
    # 3. Flatten mask
    # ---------------------------------------------------------

    mask = mask.reshape(B, num_patches)

    # (B, num_patches, 1, 1, 1)
    mask = mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

    # ---------------------------------------------------------
    # 4. Apply mask
    # ---------------------------------------------------------

    if mask_value == 0.0:
        masked_patches = patches * mask
    else:
        mask_value_tensor = torch.tensor(
            mask_value,
            dtype=images.dtype,
            device=images.device
        )

        masked_patches = torch.where(
            mask == 1.0,
            patches,
            mask_value_tensor
        )

    # ---------------------------------------------------------
    # 5. Reconstruct image
    # ---------------------------------------------------------

    masked_patches = masked_patches.view(
        B,
        num_patches_h,
        num_patches_w,
        C,
        patch_size,
        patch_size
    )

    masked_patches = masked_patches.permute(
        0, 3, 1, 4, 2, 5
    )

    masked_images = masked_patches.reshape(
        B, C, H, W
    )

    return masked_images