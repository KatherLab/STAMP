from __future__ import division
import numpy as np
from numba import  njit



def standardize_brightness(I: np.ndarray) -> np.ndarray:
    p = np.percentile(I, 90)
    return np.clip(I * 255.0 / p, 0, 255).astype(np.uint8)


def increment_zeros(I: np.ndarray) -> np.ndarray:
    """
    Remove zeros, replace with 1's.
    :param I: uint8 array
    """
    I[(I == 0)] = 1
    return I


@njit
def RGB_to_OD(I: np.ndarray) -> np.ndarray:
    """
    Convert from RGB to optical density (OD_RGB) space.

    RGB = 255 * exp(-1*OD_RGB).

    :param I: Image RGB uint8.
    :return: Optical denisty RGB image.
    """
    return np.maximum(-1 * np.log(np.maximum(I, 1) / 255), 1e-6)


@njit
def OD_to_RGB(OD: np.ndarray) -> np.ndarray:
    """
    Convert from optical density (OD_RGB) to RGB.

    RGB = 255 * exp(-1*OD_RGB)

    :param OD: Optical denisty RGB image.
    :return: Image RGB uint8.
    """
    OD = np.maximum(OD, 1e-6)
    return np.clip(255 * np.exp(-OD), 0, 255).astype(np.uint8)


def normalize_rows(A: np.ndarray) -> np.ndarray:
    """
    Normalize rows of an array
    :param A:
    :return:
    """
    return A / np.linalg.norm(A, axis=-1, keepdims=True)


@njit
def calc_hematoxylin(source_concentrations, h, w):
    H = source_concentrations[:, 0].reshape(h, w)
    H = np.exp(-H)
    return H


@njit
def get_principle_colors(OD: np.ndarray, V: np.ndarray, angular_percentile: int = 1):
    # Project OD pixels on the plane of the two principle components.
    That = OD @ V

    # Angular coordinates with respect to the principle, orthogonal eigenvectors
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, angular_percentile)
    maxPhi = np.percentile(phi, 100 - angular_percentile)

    # The two principle colors are the unit vectors with the 1- and 99-percentile
    # angular coordinates in the 2 principle component space of the slide
    v1 = V @ np.array([np.cos(minPhi), np.sin(minPhi)])
    v2 = V @ np.array([np.cos(maxPhi), np.sin(maxPhi)])
    return v1, v2


def norm_patch(
    patch_OD: np.ndarray,
    stain_matrix_src: np.ndarray,
    stain_matrix_target: np.ndarray,
    max_target_conc: np.ndarray,
    luminosity_threshold: float = 0.15,
) -> np.ndarray:
    patch_shape = patch_OD.shape
    patch_OD = patch_OD.reshape(-1, 3) # shape: (patch_h * patch_w, 3)

    # ignore background pixels during concentration calculations
    # to prevent color distortions of the background
    mask = (patch_OD > luminosity_threshold).any(axis=-1)
    patch_OD_masked = patch_OD[mask]
    
    if patch_OD_masked.shape[0] > 0:
        # calculate the source concentration
        src_conc, *_ = np.linalg.lstsq(stain_matrix_src.T, patch_OD_masked.T, rcond=None)
        src_conc = src_conc.T # shape: (patch_h * patch_w, 2)

        max_src_conc = np.percentile(src_conc, 99, axis=0) # shape: (2,)
        src_conc *= max_target_conc / max_src_conc

        # convert HE concentrations back to OD-RGB color space
        patch_OD[mask] = src_conc @ stain_matrix_target
    
    # convert to RGB color space
    patch_normed = np.clip(255 * np.exp(-patch_OD), 0, 255).astype(np.uint8)
    patch_normed = patch_normed.reshape(patch_shape)
    return patch_normed


def get_concentrations(arr: np.ndarray, stain_matrix: np.ndarray, is_OD: bool = False) -> np.ndarray:
    """
    Get concentrations, a npix x 2 matrix
    :param I:
    :param stain_matrix: a 2x3 stain matrix
    :return:
    """
    if not is_OD:
        OD = RGB_to_OD(arr).reshape(-1, 3)
    else:
        OD = arr.reshape(-1, 3)
    try:
        # stain_matrix.T @ x = OD.T
        x, *_ = np.linalg.lstsq(stain_matrix.T, OD.T, rcond=None)
        x = x.T
    except Exception as e:
        print(e)
        x = None
    return x
