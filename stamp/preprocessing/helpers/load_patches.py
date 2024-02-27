import math
import numpy as np
from PIL import Image



def reconstruct_from_patches(patches, patches_coords, img_shape):
    img_h, img_w = img_shape
    patch_h, patch_w = patches.shape[1:3]
    img = Image.new("RGB", (img_w, img_h))
    for (x, y), patch in zip(patches_coords, patches):
        img.paste(
            Image.fromarray(patch[:patch_h, :patch_w]),
            (y, x, y + patch_w, x + patch_h)
        )
    return img


def extract_patches(img, patch_size, pad=False, drop_empty=False):
    patch_size = np.array(patch_size)
    if pad:
        rows, cols = np.ceil(np.array(img.shape)[:2] / patch_size).astype(int)
    else: # if pad=False, then too small patches at the right and bottom border are getting discarded
        rows, cols = np.array(img.shape)[:2] // patch_size
    n_max = rows * cols

    
    patches = np.zeros((n_max, patch_size[0], patch_size[1], img.shape[-1]), dtype=np.uint8)
    patches_coords = np.zeros((n_max, 2), dtype=np.uint16)
    k = 0
    for i in range(rows):
        for j in range(cols):
            x, y = i*patch_size[0], j*patch_size[1]
            patch = img[x:x+patch_size[0], y:y+patch_size[1]]
            # skip empty/black patches
            if drop_empty and not patch.any():
                continue
            # pad on the left and bottom so all patches have the same size
            if pad and ((real_shape := np.array(patch.shape[:2])) < patch_size).any():
                padding = patch_size - real_shape
                patch = np.pad(patch, pad_width=((0, padding[0]), (0, padding[1]), (0, 0)), mode="mean")
            
            patches[k] = patch
            patches_coords[k] = (x, y)
            k += 1
    
    return patches[:k], patches_coords[:k], n_max
