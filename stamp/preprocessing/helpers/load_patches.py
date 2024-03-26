import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple


def extract_patches(
    img: np.ndarray, patch_size: Tuple[int, int], pad: bool = False, drop_empty: bool = False, overlap: bool = False
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Splits a the whole slide image into smaller patches.
    If `drop_empty`=True completly black patches are removed. This is useful
    when `img` was obtained from an JPEG of an already Canny+normed WSI. 
    """
    kernel_size = np.array(patch_size)
    img_size = np.array(img.shape)[:2]
    stride = kernel_size if not overlap else kernel_size // 2
    if pad:
        rows, cols = np.ceil((img_size - kernel_size) / stride + 1).astype(int)
    else:  # if pad=False, then too small patches at the right and bottom border are getting discarded
        rows, cols = (img_size - kernel_size) // stride + 1
    n_max = rows * cols

    # overestimate the number of non-empty patches
    patches = np.zeros((n_max, kernel_size[0], kernel_size[1], img.shape[-1]), dtype=np.uint8)
    # patches_coords stores the (height, width)-coordinate of the top-left corner for each patch
    patches_coords = np.zeros((n_max, 2), dtype=np.uint16)

    k = 0
    for i in range(rows):
        for j in range(cols):
            x, y = (i, j) * stride
            patch = img[x : x + kernel_size[0], y : y + kernel_size[1]]
            
            if drop_empty and not patch.any():  # skip empty/black patches
                continue
            
            # pad on the left and bottom so patches on the edges of the WSI have the same size
            if pad and ((real_shape := np.array(patch.shape[:2])) < kernel_size).any():
                padding = kernel_size - real_shape
                patch = np.pad(
                    patch,
                    pad_width=((0, padding[0]), (0, padding[1]), (0, 0)),
                    mode="mean",
                )

            patches[k] = patch
            patches_coords[k] = (x, y)
            k += 1

    return patches[:k], patches_coords[:k], n_max


def reconstruct_from_patches(
    patches: np.ndarray, patches_coords: np.array, img_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Reconstruct the WSI from the patches.
    `patches` is of shape (num_patches, patch_height, patch_width, channels)
    """
    img_h, img_w = img_shape
    patch_h, patch_w = patches.shape[1:3]
    img = Image.new("RGB", (img_w, img_h))
    for (x, y), patch in zip(patches_coords, patches):  # (x, y) = (height, width)
        img.paste(
            Image.fromarray(patch[:patch_h, :patch_w]),
            (y, x, y + patch_w, x + patch_h)
        )
    return img
