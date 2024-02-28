__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"

import os
from pathlib import Path
import logging
from contextlib import contextmanager
import time
from datetime import timedelta
from typing import Optional

import openslide
from tqdm import tqdm
from PIL import Image
import numpy as np
import torch

from .normalizer.normalizer import MacenkoNormalizer
from .extractor.feature_extractors import FeatureExtractor, store_features, store_metadata
from .helpers.common import supported_extensions
from .helpers.exceptions import MPPExtractionError
from .helpers.load_slides import load_slide
from .helpers.load_patches import extract_patches, reconstruct_from_patches
from .helpers.background_rejection import filter_background



Image.MAX_IMAGE_PIXELS = None


@contextmanager
def lock_file(slide_path: Path):
    try:
        Path(f"{slide_path}.tmp").touch()
    except PermissionError:
        pass # No write permissions for wsi directory
    try:
        yield
    finally:
        if os.path.exists(f"{slide_path}.tmp"): # Catch collision cases
            os.remove(f"{slide_path}.tmp")


def test_wsidir_write_permissions(wsi_dir: Path):
    try:
        testfile = wsi_dir/f"test_{time.time()}.tmp"
        Path(testfile).touch()
    except PermissionError:
        logging.warning("No write permissions for wsi directory! If multiple stamp processes are running "
                        "in parallel, the final summary may show an incorrect number of slides processed.")
    finally:
        if os.path.exists(testfile):
            os.remove(testfile)


def save_image(image, path: Path):
    width, height = image.size
    if width > 65500 or height > 65500:
        logging.warning(f"Image size ({width}x{height}) exceeds maximum size of 65500x65500, resizing {path.name} before saving...")
        ratio = 65500 / max(width, height)
        image = image.resize((int(width * ratio), int(height * ratio)))
    image.save(path)


def preprocess(output_dir: Path, wsi_dir: Path, model_path: Path, cache_dir: Path,
               cache: bool = False, norm: bool = False, normalization_template: Optional[Path] = None,
               del_slide: bool = False, only_feature_extraction: bool = False,
               keep_dir_structure: bool = False, cores: int = 8, target_microns: int = 256,
               patch_size: int = 224, batch_size: int = 64, device: str = "cuda"
               ):
    target_mpp = target_microns / patch_size
    patch_size = (patch_size, patch_size) # (224, 224) by default

    # Initialize the feature extraction model
    print(f"Initializing CTransPath model as feature extractor...")
    has_gpu = torch.cuda.is_available()
    device = torch.device(device) if "cuda" in device and has_gpu else torch.device("cpu")
    extractor = FeatureExtractor.from_checkpoint(checkpoint_path=model_path, device=device)

    # Create output and cache directories
    output_dir.mkdir(parents=True, exist_ok=True)
    norm_method = "STAMP_macenko_" if norm else "STAMP_raw_"
    model_name_norm = Path(norm_method + extractor.model_name)
    output_file_dir = output_dir/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)
    if cache:
        cache_dir.mkdir(exist_ok=True, parents=True)

    # Create logfile and set up logging
    logfile_name = "logfile_" + time.strftime("%Y-%m-%d_%H-%M-%S")
    logdir = output_file_dir/logfile_name
    logging.basicConfig(filename=logdir, force=True, level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info("Preprocessing started at: " + time.strftime("%Y-%m-%d %H:%M:%S"))
    logging.info(f"Norm: {norm} | Target_microns: {target_microns} | Patch_size: {patch_size} | MPP: {target_mpp}")
    logging.info(f"Model: {extractor.model_name}\n")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Stored logfile in {logdir}")
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {cores}")
    print(f"GPU is available: {has_gpu}")
    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")
    test_wsidir_write_permissions(wsi_dir)

    if norm:
        assert normalization_template is not None, "`normalization_template` can't be None if `norm`=True"
        print("\nInitializing Macenko normalizer...")
        normalizer = MacenkoNormalizer()
        print(f"Reference: {normalization_template}")
        target = Image.open(normalization_template).convert('RGB')
        normalizer.fit(np.array(target))  

    total_start_time = time.time()
    img_name = "norm_slide.jpg" if norm else "canny_slide.jpg"

    # Get list of slides, filter out slides that have already been processed
    print("Scanning for existing feature files...")
    existing = [f.stem for f in output_file_dir.glob("**/*.h5")] if output_file_dir.exists() else []
    if not only_feature_extraction:
        img_dir = [svs for ext in supported_extensions for svs in wsi_dir.glob(f"**/*{ext}")]
        existing = [f for f in existing if f in [f.stem for f in img_dir]]
        img_dir = [f for f in img_dir if f.stem not in existing]
    else:
        if not cache_dir.exists():
            logging.error("Cache directory does not exist, cannot extract features from cached slides!")
            exit(1)
        img_dir = [jpg for jpg in cache_dir.glob(f"**/*/{img_name}")]
        existing = [f for f in existing if f in [f.parent.name for f in img_dir]]
        img_dir = [f for f in img_dir if f.parent.name not in existing]

    num_processed, num_total = 0, len(img_dir) + len(existing)
    error_slides = []
    if len(existing):
        print(f"For {len(existing)} out of {num_total} slides in the wsi directory feature files were found, skipping these slides...")
    for slide_url in tqdm(img_dir, "\nPreprocessing progress", leave=False, miniters=1, mininterval=0):
        slide_name = slide_url.stem if not only_feature_extraction else slide_url.parent.name
        slide_cache_dir = cache_dir/slide_name
        if cache:
            slide_cache_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"\n\n===== Processing slide {slide_name} =====")
        slide_subdir = slide_url.parent.relative_to(wsi_dir)
        if not keep_dir_structure or slide_subdir == Path("."):
            feat_out_dir = output_file_dir/slide_name
        else:
            (output_file_dir/slide_subdir).mkdir(parents=True, exist_ok=True)
            feat_out_dir = output_file_dir/slide_subdir/slide_name
        if not (os.path.exists((f"{feat_out_dir}.h5"))) and not os.path.exists(f"{slide_url}.tmp"):
            with lock_file(slide_url):
                if (
                    (only_feature_extraction and (slide_jpg := slide_url).exists()) or \
                    (slide_jpg := slide_cache_dir/"norm_slide.jpg").exists()
                ):
                    slide_array = np.array(Image.open(slide_jpg))
                    patches, patches_coords, n = extract_patches(slide_array, patch_size, pad=False, drop_empty=True)
                    print(f"Loaded {img_name}, {patches.shape[0]}/{n} tiles remain")
                    # note that due to being stored as an JPEG rejected patches which
                    # neighbor accepted patches will most likely also be loaded
                    # thus we again apply a background filtering
                    patches, patches_coords = filter_background(patches, patches_coords, cores)
                    # patches.shape = (n_patches, patch_h, patch_w, 3)
                    # patches_coords.shape = (n_patches, 2)
                else:
                    try:
                        slide = openslide.OpenSlide(slide_url)
                    except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                        logging.error("Unsupported format for slide, continuing...")
                        error_slides.append(slide_name)
                        continue
                    except Exception as e:
                        logging.error(f"Failed loading slide, continuing... Error: {e}")
                        error_slides.append(slide_name)
                        continue

                    start_loading = time.time()
                    try:
                        slide_array = load_slide(slide=slide, target_mpp=target_mpp, cores=cores)
                    except MPPExtractionError:
                        if del_slide:
                            logging.error("MPP missing in slide metadata, deleting slide and continuing...")
                            if os.path.exists(slide_url):
                                os.remove(slide_url)
                        else:
                            logging.error("MPP missing in slide metadata, continuing...")
                        error_slides.append(slide_name)
                        continue
                    except openslide.lowlevel.OpenSlideError as e:
                        print("")
                        logging.error(f"Failed loading slide, continuing... Error: {e}")
                        error_slides.append(slide_name)
                        continue
                    del slide   # Remove .SVS from memory

                    print(f" Loaded slide ({time.time() - start_loading:.2f} seconds)")
                    print(f"Size of WSI: {slide_array.shape}")
                        
                    if cache:   # Save raw .svs jpg
                        raw_image = Image.fromarray(slide_array)
                        save_image(raw_image, slide_cache_dir/"slide.jpg")

                    # Canny edge detection to discard tiles containing no tissue BEFORE normalization
                    patches, patches_coords, _ = extract_patches(slide_array, patch_size, pad=False, drop_empty=True)
                    patches, patches_coords = filter_background(patches, patches_coords, cores)
                    # patches.shape = (n_patches, patch_h, patch_w, 3)
                    # patches_coords.shape = (n_patches, 2)

                    if cache:
                        print("Saving Canny background rejected image...")
                        canny_img = reconstruct_from_patches(patches, patches_coords, slide_array.shape[:2])
                        save_image(canny_img, slide_cache_dir/"canny_slide.jpg")

                    # Pass raw slide_array for getting the initial concentrations, tissue_patches for actual normalization
                    if norm:
                        print(f"\nNormalizing slide...")
                        start_normalizing = time.time()                        
                        patches = normalizer.transform(slide_array, patches)                        
                        print(f"Normalized slide ({time.time() - start_normalizing:.2f} seconds)")
                        if cache:
                            norm_img = reconstruct_from_patches(patches, patches_coords, slide_array.shape[:2])
                            save_image(norm_img, slide_cache_dir/"norm_slide.jpg")

                    # Remove original slide jpg from memory
                    del slide_array
                    
                    # Optionally remove the original slide from harddrive
                    if del_slide:
                        print("Deleting slide from local folder...")
                        if os.path.exists(slide_url):
                            os.remove(slide_url)

                print("\nExtracting CTransPath features from slide...")
                start_time = time.time()
                if len(patches) > 0:
                    store_metadata(
                        outdir=feat_out_dir,
                        extractor_name=extractor.name,
                        patch_size=patch_size,
                        target_microns=target_microns,
                        normalized=norm
                    )
                    features = extractor.extract(patches, cores, batch_size)
                    store_features(feat_out_dir, features, patches_coords, extractor.name)
                    logging.info(f" Extracted features from slide: {time.time() - start_time:.2f} seconds ({features.shape[0]} tiles)")
                    num_processed += 1
                else:
                    logging.error(" 0 tiles remain to extract features from after pre-processing. Continuing...")
                    error_slides.append(slide_name)
                    continue
        else:
            if os.path.exists((f"{feat_out_dir}.h5")):
                logging.info(".h5 file for this slide already exists. Skipping...")
            else:
                logging.info("Slide is already being processed. Skipping...")
            existing.append(slide_name)
            if del_slide:
                print("Deleting slide from local folder...")
                if os.path.exists(slide_url):
                    os.remove(slide_url)

    logging.info(f"\n===== End-to-end processing time of {num_total} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} =====")
    logging.info(f"Summary: Processed {num_processed} slides, encountered {len(error_slides)} errors, skipped {len(existing)} readily-processed slides")
    if len(error_slides):
        logging.info("The following slides were not processed due to errors:\n  " + "\n  ".join(error_slides))
