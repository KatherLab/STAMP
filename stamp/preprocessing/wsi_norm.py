__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"


import argparse
from pathlib import Path
from contextlib import contextmanager
import logging
import os
import openslide
from tqdm import tqdm
import PIL
import cv2
import time
from datetime import timedelta
from pathlib import Path
import torch
from typing import Optional
from .helpers import stainNorm_Macenko
from .helpers.common import supported_extensions
from .helpers.concurrent_canny_rejection import reject_background
from .helpers.loading_slides import process_slide_jpg, load_slide, get_raw_tile_list
from .helpers.feature_extractors import FeatureExtractor, extract_features_
from .helpers.exceptions import MPPExtractionError


PIL.Image.MAX_IMAGE_PIXELS = None

@contextmanager
def lock_file(slide_url):
    Path(f'{slide_url}.tmp').touch()
    try:
        yield
    finally:
        if os.path.exists(f'{slide_url}.tmp'): # Catch collision cases
            os.remove(f'{str(slide_url)}.tmp')

def preprocess(output_dir: Path, wsi_dir: Path, model_path: Path, cache_dir: Path,
               norm: bool, del_slide: bool, only_feature_extraction: bool, 
               cores: int = 8, target_microns: int = 256, patch_size: int = 224, 
               device: str = "cuda", normalization_template: Path = None):    
    has_gpu = torch.cuda.is_available()
    target_mpp = target_microns/patch_size
    patch_shape = (patch_size, patch_size) #(224, 224) by default
    step_size = patch_size #have 0 overlap by default

    cache_dir.mkdir(exist_ok=True, parents=True)
    logfile_name = 'logfile_' + time.strftime("%Y%m%d-%H%M%S")
    logdir = cache_dir/logfile_name
    logging.basicConfig(filename=logdir, force=True, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Preprocessing started at: " + time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logging.info(f"Norm: {norm} | Target_microns: {target_microns} | Patch_size: {patch_size} | MPP: {target_mpp}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Stored logfile in {logdir}")
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {cores}")
    print(f"GPU is available: {has_gpu}")

    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")

    if norm:
        print("\nInitialising Macenko normaliser...")
        print(normalization_template)
        target = cv2.imread(str(normalization_template))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        normalizer = stainNorm_Macenko.Normalizer()
        normalizer.fit(target)

    # Initialize the feature extraction model
    print(f"\nInitialising CTransPath model as feature extractor...")
    extractor = FeatureExtractor()
    model, model_name = extractor.init_feat_extractor(checkpoint_path=model_path, device=device)
    logging.info(f"Model: {model_name}\n")

    # Create output feature folder, f.e.:
    # ~/output_folder/E2E_macenko_xiyuewang-ctranspath/
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_method = "STAMP_macenko_" if norm else "STAMP_raw_"
    model_name_norm = Path(norm_method + model_name)
    output_file_dir = output_dir/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)
    
    total_start_time = time.time()
    
    img_name = "norm_slide.jpg" if norm else "canny_slide.jpg"
    if not only_feature_extraction:
        img_dir = sum((list(wsi_dir.glob(f'**/*{ext}'))
                    for ext in supported_extensions),
                    start=[])
    else:
        img_dir = list(wsi_dir.glob(f'**/*/{img_name}'))

    for slide_url in tqdm(img_dir, "\nPreprocessing progress", leave=False, miniters=1, mininterval=0):
        if not only_feature_extraction:
            slide_name = Path(slide_url).stem
            slide_cache_dir = cache_dir/slide_name
            slide_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            slide_name = Path(slide_url).parent.name

        print("\n")
        logging.info(f"===== Processing slide {slide_name} =====")
        feat_out_dir = output_file_dir/slide_name
        if not (os.path.exists((f'{feat_out_dir}.h5'))) and not os.path.exists(f'{slide_url}.tmp'):
            with lock_file(slide_url):
                # Load WSI as one image
                if (only_feature_extraction and (slide_jpg := slide_url).exists()) \
                    or (slide_jpg := slide_cache_dir/'norm_slide.jpg').exists():
                    canny_norm_patch_list, coords_list, patch_saved, total = process_slide_jpg(slide_jpg)
                    print(f"Loaded {img_name}, {patch_saved}/{total} tiles remain")
                    if patch_saved == 0:
                        print("No tiles remain for {slide_name}, skipping...")
                        continue
                else:
                    try:
                        slide = openslide.OpenSlide(str(slide_url))
                    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                        logging.error(f"Unsupported format for {slide_name}")
                        continue
                    except Exception as e:
                        logging.error(f"Failed loading {slide_name}, error: {e}")
                        continue

                    start_time = time.time()
                    try:
                        slide_array = load_slide(slide=slide, target_mpp=target_mpp, cores=cores)
                    except MPPExtractionError:
                        if del_slide:
                            logging.error(f"Skipping slide and deleting due to missing MPP...")
                            os.remove(str(slide_url))
                        else:
                            logging.error(f"Skipping slide due to missing MPP...")
                        continue

                    # Save raw .svs jpg
                    PIL.Image.fromarray(slide_array).save(f'{slide_cache_dir}/slide.jpg')

                    # Remove .SVS from memory
                    del slide                    
                    print(f"\nLoaded slide: {time.time() - start_time:.2f} seconds")

                    #Do edge detection here and reject unnecessary tiles BEFORE normalisation
                    bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = slide_array, patch_size=patch_shape, step=step_size,
                                                                                        outdir=cache_dir, save_tiles=False, cores=cores)

                    start_time = time.time()
                    # Pass raw slide_array for getting the initial concentrations, bg_reject_array for actual normalisation
                    if norm:
                        print(f"Normalising slide...")
                        canny_img, img_norm_wsi_jpg, canny_norm_patch_list, coords_list = normalizer.transform(slide_array, bg_reject_array, 
                                                                                                            rejected_tile_array, patch_shapes, cores=cores)
                        print(f"\nNormalised slide: {time.time() - start_time:.2f} seconds")
                        img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg
                    else:
                        canny_img, canny_norm_patch_list, coords_list = get_raw_tile_list(slide_array.shape, bg_reject_array,
                                                                                        rejected_tile_array, patch_shapes)

                    print("Saving Canny background rejected image...")
                    canny_img.save(f'{slide_cache_dir}/canny_slide.jpg')
                    
                    # Remove original slide jpg from memory
                    del slide_array
                    
                    # Optionally removing the original slide from harddrive
                    if del_slide:
                        print(f"Deleting slide from local folder...")
                        os.remove(str(slide_url))

                print(f"\nExtracting CTransPath features from slide...")
                start_time = time.time()
                if len(canny_norm_patch_list) > 0:
                    extract_features_(model=model, model_name=model_name, norm_wsi_img=canny_norm_patch_list,
                                    coords=coords_list, wsi_name=slide_name, outdir=feat_out_dir, cores=cores, is_norm=norm, device=device if has_gpu else "cpu")
                    logging.info(f"Extracted features from slide: {time.time() - start_time:.2f} seconds")
                else:
                    logging.error("0 tiles remain to extract features from after pre-processing {slide_name}, skipping...")
                    continue
        else:
            if os.path.exists((f'{feat_out_dir}.h5')):
                logging.info(f".h5 file for this slide already exists. Skipping...")
            else:
                logging.info("Slide is already being processed. Skipping...")
            if del_slide:
                print("Deleting slide from local folder...")
                os.remove(str(slide_url))

    logging.info(f"===== End-to-end processing time of {len(img_dir)} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} =====")
