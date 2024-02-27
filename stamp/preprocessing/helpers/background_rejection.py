import time
from typing import Dict, Tuple, List, Any
from concurrent import futures
import numpy as np
import cv2
import PIL


def canny_fcn(patch: np.array) -> Tuple[np.array, bool]:
    h, w = patch.shape[:2]
    patch_img = PIL.Image.fromarray(patch)
    patch_gray = np.array(patch_img.convert('L'))
    # tile_to_grayscale is an PIL.Image.Image with image mode L
    # Note: If you have an L mode image, that means it is
    # a single channel image - normally interpreted as grayscale.
    # The L means that is just stores the Luminance.
    # It is very compact, but only stores a grayscale, not color.

    # hardcoded thresholds
    edge = cv2.Canny(patch_gray, 40, 100)
    edge = (edge / np.max(edge)) if np.max(edge) != 0 else 0    # avoid dividing by zero
    edge = (np.sum(np.sum(edge)) / (h * w) * 100) if (h * w) != 0 else 0

    # hardcoded limit. Less or equal to 2 edges will be rejected (i.e., not saved)
    return edge < 2


def filter_background(patches: np.array, patches_coords: np.array, cores: int = 8) -> \
Tuple[np.ndarray, np.ndarray, List[Any]]:
    # patch_shape = np.array(patch_shape)
    # h_patches, w_patches = np.ceil(np.array(img.shape)[:2] / patch_shape).astype(int)
    n = len(patches)
    print(f"\nCanny background rejection...")
    
    begin = time.time()
    has_tissue = np.zeros((n,), dtype=bool)
    with futures.ThreadPoolExecutor(cores) as executor:
        future_coords: Dict[futures.Future, int] = {}
        for k, patch in enumerate(patches):
            future = executor.submit(canny_fcn, patch)
            future_coords[future] = k
        
        #num of patches x 224 x 224 x 3 for RGB patches
        # tissue_patches = np.zeros((n, patch_shape[0], patch_shape[1], 3), dtype=np.uint8)
        # has_tissue = np.zeros(n, dtype=bool)
        for tile_future in futures.as_completed(future_coords):
            k = future_coords[tile_future]
            is_rejected = tile_future.result()
            has_tissue[k] = not is_rejected
        
    patches = patches[has_tissue]
    patches_coords = patches_coords[has_tissue]

    print(f"Finished Canny background rejection, rejected {np.sum(~has_tissue)}/{n} tiles ({time.time()-begin:.2f} seconds)")
    return patches, patches_coords
