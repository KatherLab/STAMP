"""
Stain normalization based on the method of:

M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

For the original implementation see: https://github.com/Peter554/StainTools
"""
from __future__ import division
import time
from concurrent import futures
import numpy as np
from tqdm import tqdm

from . import utils


class MacenkoNormalizer:
    """
    A stain normalization using the methods proposed by Macenko et al.
    """
    def __init__(self):
        self.stain_matrix_target = None
        self.max_target_conc = None

    def fit(self, target):
        target = utils.standardize_brightness(target)
        target_OD = utils.RGB_to_OD(target)
        self.stain_matrix_target = self.get_stain_matrix(target_OD, undersample=False) # shape: (2, 3)
        target_conc = utils.get_concentrations(target_OD, self.stain_matrix_target, is_OD=True)
        self.max_target_conc = np.percentile(target_conc, 99, axis=0) # shape: (2,)
        return self
        
    def transform(self, patches: np.ndarray, cores: int = 8) -> np.ndarray:
        """Returns an array the same shape as `patches` with Macenko normalization applied to all patches."""
        start_normalizing = time.time()
        luminosity_threshold: float = 0.15

        # skip normalization if no patches are given
        if patches.shape[0] == 0:
            return patches

        # convert from RGB to optical density (OD_RGB) space
        patches_OD = utils.RGB_to_OD(patches) # shape: (n_patches, patch_h, patch_w, 3)

        # calculates the stain matrix only from the patches that contain some tissue
        stain_matrix_src = self.get_stain_matrix(patches_OD, luminosity_threshold, undersample=True) # shape: (2, 3)
        print(f"Get stain matrix ({(after_stain_mat := time.time()) - start_normalizing:.2f} seconds)")

        norm_patches = self._norm_patches_threaded(patches_OD, stain_matrix_src, luminosity_threshold, cores)
        # shape: (n_patches, patch_h, patch_w, 3)
        print(f" Normalized {len(patches)} patches ({time.time()-after_stain_mat:.2f} seconds)")

        return norm_patches
        
    def _norm_patches_threaded(
        self,
        patches_OD: np.ndarray,
        stain_matrix_src: np.ndarray,
        luminosity_threshold: float = 0.15,
        cores: int = 8,
    ) -> np.ndarray:
        patches_shape = patches_OD.shape
        with futures.ThreadPoolExecutor(cores) as executor:
            future_coords: dict[futures.Future, int] = {}
            for i, patch in enumerate(patches_OD):
                future = executor.submit(
                    utils.norm_patch,
                    patch,
                    stain_matrix_src,
                    self.stain_matrix_target,
                    self.max_target_conc,
                    luminosity_threshold
                )
                future_coords[future] = i

            norm_patches = np.zeros(patches_shape, dtype=np.uint8)
            for tile_future in tqdm(
                futures.as_completed(future_coords),
                total=patches_shape[0],
                desc="Normalizing patches",
                leave=False,
            ):
                i = future_coords[tile_future]
                patch = tile_future.result()
                norm_patches[i] = patch
        return norm_patches

    @staticmethod
    def get_stain_matrix(
        OD: np.ndarray, luminosity_threshold: float = 0.15, angular_percentile: int = 1, undersample: bool = False
    ) -> np.ndarray:
        """
        Stain matrix estimation via method of:
        M. Macenko et al. 'A method for normalizing histology slides for quantitative analysis'

        :param I: Image RGB uint8.
        :param luminosity_threshold:
        :param angular_percentile:
        :return:
        """
        if undersample: # alternatively Poisson disk?
            OD = OD[::3, ::3] # only use ~11% of the data per patch
        OD = OD.reshape(-1, 3)

        # Optional additional background filtering
        if luminosity_threshold > 0:
            OD = OD[(OD > luminosity_threshold).any(axis=-1)]

        # Eigenvectors of cov in OD space (orthogonal as cov symmetric)
        V = np.linalg.eigh(np.cov(OD, rowvar=False)).eigenvectors

        # The two principle eigenvectors
        V = V[:, [2, 1]]

        # Make sure vectors are pointing the right way
        V[:, 0] *= np.sign(V[0, 0])
        V[:, 1] *= np.sign(V[0, 1])

        # the two "principle" colors
        v1, v2 = utils.get_principle_colors(OD, V, angular_percentile)

        # Order of H and E. H first row.
        if v1[0] > v2[0]:
            HE = np.array([v1, v2])
        else:
            HE = np.array([v2, v1])
        return utils.normalize_rows(HE)
    
    def target_stains(self):
        return utils.OD_to_RGB(self.stain_matrix_target)

    def hematoxylin(self, I: np.ndarray, use_target_stain: bool = False) -> np.ndarray:
        """
        Extracts the hematoxylin stain from the input image.
        If `use_target_stain=True`, adjusts the RGB values of the hematoxylin stain
        to match a target images stain. Defaults to False.
        """
        # I = utils.standardize_brightness(I)
        OD = utils.RGB_to_OD(I)
        stain_matrix_source = self.get_stain_matrix(OD)
        src_concs = utils.get_concentrations(OD, stain_matrix_source, is_OD=True)
        hema_conc = src_concs[:, :1]

        if use_target_stain:
            # Scale the concentration of hematoxylin to match a target maximal concentration
            max_hema_conc = np.percentile(hema_conc, 99, axis=0)
            hema_conc *= self.max_target_conc[0] / max_hema_conc
        
        if use_target_stain:
            hema_OD_RGB = self.stain_matrix_target[0]
        else:
            hema_OD_RGB = stain_matrix_source[0]

        # Convert concentration first to OD-RGB space and then TO RGB space
        hematoxylin = hema_conc * hema_OD_RGB
        hematoxylin = utils.OD_to_RGB(hematoxylin)
        return hematoxylin.reshape(I.shape)
        
    def eosin(self, I: np.ndarray, use_target_stain: bool = False) -> np.ndarray:
        """
        Extracts the eosin stain from the input image.
        If `use_target_stain=True`, adjusts the RGB values of the eosin stain
        to match a target images stain. Defaults to False.
        """
        # I = utils.standardize_brightness(I)
        OD = utils.RGB_to_OD(I)
        stain_matrix_source = self.get_stain_matrix(OD)
        src_concs = utils.get_concentrations(OD, stain_matrix_source, is_OD=True)
        eosin_conc = src_concs[:, 1:2]

        if use_target_stain:
            # Scale the concentration of eosin to match a target maximal concentration
            max_eosin_conc = np.percentile(eosin_conc, 99, axis=0)
            eosin_conc *= self.max_target_conc[1] / max_eosin_conc
        
        if use_target_stain:
            eosin_OD_RGB = self.stain_matrix_target[1]
        else:
            eosin_OD_RGB = stain_matrix_source[1]
            
        # Convert concentration first to OD-RGB space and then TO RGB space
        eosin = eosin_conc * eosin_OD_RGB
        eosin = utils.OD_to_RGB(eosin)
        return eosin.reshape(I.shape)
