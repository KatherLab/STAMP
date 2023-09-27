"""
Stain normalization based on the method of:

M. Macenko et al., ‘A method for normalizing histology slides for quantitative analysis’, in 2009 IEEE International Symposium on Biomedical Imaging: From Nano to Macro, 2009, pp. 1107–1110.

Uses the spams package:

http://spams-devel.gforge.inria.fr/index.html

Use with python via e.g https://anaconda.org/conda-forge/python-spams
"""

from __future__ import division

import numpy as np
import helpers.stain_utils as ut
from numba import njit
import time
from concurrent import futures
from tqdm import tqdm
from typing import Dict, Tuple
import os

@njit
def v1v2_mult(V, minPhi, maxPhi):
    v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    return v1, v2

def get_stain_matrix(I, beta=0.15, alpha=1):
    """
    Get stain matrix (2x3)
    :param I:
    :param beta:
    :param alpha:
    :return:
    """
    OD = ut.RGB_to_OD(I).reshape((-1, 3))
    OD = (OD[(OD > beta).any(axis=1), :])
    _, V = np.linalg.eigh(np.cov(OD, rowvar=False))
    V = V[:, [2, 1]]
    if V[0, 0] < 0: V[:, 0] *= -1
    if V[0, 1] < 0: V[:, 1] *= -1
    That = np.dot(OD, V)
    phi = np.arctan2(That[:, 1], That[:, 0])
    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)
    v1, v2 = v1v2_mult(V, minPhi, maxPhi)

    if v1[0] > v2[0]:
        HE = np.array([v1, v2])
    else:
        HE = np.array([v2, v1])
    # v1 = np.dot(V, np.array([np.cos(minPhi), np.sin(minPhi)]))
    # v2 = np.dot(V, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
    # if v1[0] > v2[0]:
    #     HE = np.array([v1, v2])
    # else:
    #     HE = np.array([v2, v1])
    return ut.normalize_rows(HE)


###
@njit
def transform_return(source_concentrations, stain_matrix_target, maxC_target, maxC_source, patch_shape):
    source_concentrations *= (maxC_target / maxC_source)
    return (255 * np.exp(-1 * np.dot(source_concentrations, stain_matrix_target).reshape(patch_shape))).astype(
        np.uint8)



@njit
def hematoxalin_return(source_concentrations, h, w):
    H = source_concentrations[:, 0].reshape(h, w)
    H = np.exp(-1 * H)
    return H

def concurrent_concXstain(self, source_concentrations, patch_shapes, idx):
    maxC_source = np.percentile(source_concentrations, 99, axis=0).reshape((1, 2))
    maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
    jit_output = transform_return(source_concentrations, self.stain_matrix_target, maxC_target, maxC_source, patch_shapes[idx]) 
    return(jit_output)

from PIL import Image
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

class Normalizer(object):
    """
    A stain normalization object
    """

    def __init__(self):
        self.stain_matrix_target = None
        self.target_concentrations = None

    def fit(self, target):
        target = ut.standardize_brightness(target)
        self.stain_matrix_target = get_stain_matrix(target)
        self.target_concentrations = ut.get_concentrations_target(target, self.stain_matrix_target)

    def target_stains(self):
        return ut.OD_to_RGB(self.stain_matrix_target)


    def transform(self, og_img: np.array, bg_rejected_img: np.array, rejected_list: np.array, patch_shapes: list, cores: int=8): #TODO: add optional split, patch sizes, overlap
        begin = time.time()
        I = ut.standardize_brightness(og_img)
        after_sb = time.time()
        print(f'Standardized brightness: {after_sb-begin}')
        stain_matrix_source = get_stain_matrix(I)
        after_sm = time.time()
        print(f'Get stain matrix: {after_sm-begin}')
        I_shape = I.shape
        source_concentrations_list = ut.get_concentrations_source(bg_rejected_img, I_shape, stain_matrix_source, rejected_list)
        after_conc = time.time()
        print(f'\nGet concentrations (normalisation): {after_conc-after_sm}')

        del I, stain_matrix_source

        split=True
        if split:
            #added concurent concentrations x stain matrix
            with futures.ThreadPoolExecutor(cores) as executor:
                future_coords: Dict[futures.Future, int] = {}

                for i, source_concentrations in enumerate(source_concentrations_list):
                    # if all zeroes, skip
                    if np.any(source_concentrations):
                        future = executor.submit(
                        concurrent_concXstain, self, source_concentrations=source_concentrations, patch_shapes=patch_shapes, idx=i)
                        future_coords[future] = i
                
                norm_img_patches_list = np.zeros((len(source_concentrations_list), 224, 224, 3), dtype=np.uint8)
                for tile_future in tqdm(futures.as_completed(future_coords), total=len(source_concentrations_list)-len(rejected_list), desc='Concentrations x Stain', leave=False):
                    i = future_coords[tile_future]
                    patch = tile_future.result()
                    norm_img_patches_list[i] = patch
                
            after_transform = time.time()
            print(f'Concentrations x Stain matrix: {after_transform-after_conc}')

            print('Reconstructing image from patches...')
            norm_output_array = []
            canny_output_array = []
            for i in range(len(norm_img_patches_list)):
                # patch_shape = norm_img_patches_list[i].shape
                norm_output_array.append(np.array(norm_img_patches_list[i])) #removed .reshape(patch_shapes[i])
                if not rejected_list[i]:
                    canny_output_array.append(np.array(norm_img_patches_list[i])) #removed .reshape(patch_shapes[i])

            output_img = Image.new("RGB", (I_shape[1], I_shape[0]))
            canny_img = Image.new("RGB", (I_shape[1], I_shape[0]))

            coords_list=[]
            i_range = range(I_shape[0]//patch_shapes[0][0])
            j_range = range(I_shape[1]//patch_shapes[0][1])
            for i in i_range:
                for j in j_range:
                    idx = i*len(j_range) + j
                    output_img.paste(Image.fromarray(np.array(norm_output_array[idx])), (j*patch_shapes[idx][1], 
                                                                                i*patch_shapes[idx][0], 
                                                                                j*patch_shapes[idx][1]+patch_shapes[idx][1], 
                                                                                i*patch_shapes[idx][0]+patch_shapes[idx][0]))
                    canny_img.paste(Image.fromarray(np.array(bg_rejected_img[idx])), (j*patch_shapes[idx][1], 
                                                                                i*patch_shapes[idx][0], 
                                                                                j*patch_shapes[idx][1]+patch_shapes[idx][1], 
                                                                                i*patch_shapes[idx][0]+patch_shapes[idx][0]))
                    # print((j*patch_shapes[idx][1], 
                    #                                                             i*patch_shapes[idx][0], 
                    #                                                             j*patch_shapes[idx][1]+patch_shapes[idx][1], 
                    #                                                             i*patch_shapes[idx][0]+patch_shapes[idx][0]))
                    if not rejected_list[idx]:
                        coords_list.append((j*patch_shapes[idx][1], i*patch_shapes[idx][0]))


            #output_img = np.uint8(reconstruct_from_patches_2d(np.array(output), I_shape))
            del norm_img_patches_list
            # output_img = output_patches_list.reshape(1,-1)
            # breakpoint()
        
        else:
            maxC_source = np.percentile(source_concentrations_list, 99, axis=0).reshape((1, 2))
            maxC_target = np.percentile(self.target_concentrations, 99, axis=0).reshape((1, 2))
            jit_output = transform_return(source_concentrations_list, self.stain_matrix_target, maxC_target, maxC_source, I_shape) #I_shape, @3 (removed)
            after_transform = time.time()
            print(f'Concentrations x Stain matrix: {after_transform-after_conc}')
            output_img = Image.fromarray(np.array(jit_output))
            output_array = jit_output
            coords_list = None #TODO

        return canny_img, output_img, canny_output_array, coords_list


    def hematoxylin(self, I):
        I = ut.standardize_brightness(I)
        h, w, c = I.shape
        stain_matrix_source = get_stain_matrix(I)
        source_concentrations = ut.get_concentrations_target(I, stain_matrix_source) #put target here, just in case

        del I
        del stain_matrix_source

        jit_output = hematoxalin_return(source_concentrations, h, w)

        return jit_output

#test get_stain_matrix
def test_get_stain_matrix():
    img = Image.open('test_images/1.jpg')
    img = np.array(img)
    img = ut.standardize_brightness(img)
    stain_matrix = get_stain_matrix(img)
    assert stain_matrix.shape == (2,3)
    assert np.allclose(stain_matrix, [[0.650, 0.072, 0.740], [0.704, 0.990, 0.080]], atol=0.01)
