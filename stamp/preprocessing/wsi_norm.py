__author__ = "Omar El Nahhas"
__copyright__ = "Copyright 2023, Kather Lab"
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Omar"]
__email__ = "omar.el_nahhas@tu-dresden.de"


import argparse
from pathlib import Path
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

def preprocess(output_dir: Path, wsi_dir: Path, model_path: Path, cache_dir: Path, patch_size: int, target_mpp: float, cores: int, norm: bool, del_slide: bool, only_feature_extraction: bool, device: str = "cpu", normalization_template: Path = None):
    ### START INITIALIZATION
    print(f"Current working directory: {os.getcwd()}")
    Path(cache_dir).mkdir(exist_ok=True, parents=True)
    logdir = cache_dir/'logfile'
    logging.basicConfig(filename=logdir, force=True)
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(f'Stored logfile in {logdir}')
    #init the Macenko normaliser
    print(f"Number of CPUs in the system: {os.cpu_count()}")
    print(f"Number of CPU cores used: {cores}")
    has_gpu=torch.cuda.is_available()
    print(f"GPU is available: {has_gpu}")
    patch_shape = (patch_size, patch_size) #(224, 224) by default
    step_size = patch_size #have 0 overlap by default

    if has_gpu:
        print(f"Number of GPUs in the system: {torch.cuda.device_count()}, using device {device}")

    if norm:
        print("\nInitialising Macenko normaliser...")
        if not normalization_template:
            from ..utils import DEFAULT_RESOURCES_DIR as normalization_template
        print(normalization_template)
        target = cv2.imread(str(normalization_template))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        normalizer = stainNorm_Macenko.Normalizer()
        normalizer.fit(target)
        logging.info('Running WSI to normalised feature extraction...')
    else:
        logging.info('Running WSI to raw feature extraction...')

    #initialize the feature extraction model
    print(f"\nInitialising CTransPath model as feature extractor...")
    extractor = FeatureExtractor()
    model, model_name = extractor.init_feat_extractor(checkpoint_path=model_path)

    #create output feature folder, f.e.:
    #~/output_folder/E2E_macenko_xiyuewang-ctranspath/
    (output_dir).mkdir(parents=True, exist_ok=True)
    
    norm_method = "STAMP_macenko_" if norm else "STAMP_raw_"
    model_name_norm = Path(norm_method+model_name)
    output_file_dir = output_dir/model_name_norm
    output_file_dir.mkdir(parents=True, exist_ok=True)
    ### END INITIALIZATION
    
    total_start_time = time.time()
    
    img_name = "norm_slide.jpg" if norm else "canny_slide.jpg"
    if not only_feature_extraction:
        img_dir = sum((list(wsi_dir.glob(f'**/*.{ext}'))
                    for ext in supported_extensions),
                    start=[])
    else:
        img_dir = list(wsi_dir.glob(f'**/*/{img_name}'))
                       
    for slide_url in (progress := tqdm(img_dir, leave=False)):
        
        if not only_feature_extraction:
            slide_name = Path(slide_url).stem
            slide_cache_dir = cache_dir/slide_name
            slide_cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            slide_name = Path(slide_url).parent.name

        progress.set_description(slide_name)
        
        feat_out_dir = output_file_dir/slide_name

        if not (os.path.exists((f'{feat_out_dir}.h5'))):
            # Load WSI as one image
            if (only_feature_extraction and (slide_jpg := slide_url).exists()) \
                or (slide_jpg := slide_cache_dir/'norm_slide.jpg').exists():
                canny_norm_patch_list, coords_list, patch_saved, total = process_slide_jpg(slide_jpg)
                print(f"Loaded {img_name}, {patch_saved}/{total} tiles remain")
                if patch_saved == 0:
                    print("No tiles remain for {slide_name}, skipping...")
                    continue
            else:
                logging.info(f"\nLoading {slide_name}")
                try:
                    slide = openslide.OpenSlide(str(slide_url))
                except openslide.lowlevel.OpenSlideUnsupportedFormatError:
                    logging.error(f"Unsupported format for {slide_name}")
                    continue
                except Exception as e:
                    logging.error(f"Failed loading {slide_name}, error: {e}")
                    continue
 
                #measure time performance
                start_time = time.time()
                try:
                    slide_array = load_slide(slide=slide, target_mpp=target_mpp, cores=cores)
                except MPPExtractionError:
                    if del_slide:
                        print(f"Skipping slide and deleting {slide_url} due to missing MPP...")
                        os.remove(str(slide_url))
                    continue

                #save raw .svs jpg
                (PIL.Image.fromarray(slide_array)).save(f'{slide_cache_dir}/slide.jpg')

                #remove .SVS from memory
                del slide
                
                print("\n--- Loaded slide: %s seconds ---" % (time.time() - start_time))
                #########################

                #########################
                #Do edge detection here and reject unnecessary tiles BEFORE normalisation
                bg_reject_array, rejected_tile_array, patch_shapes = reject_background(img = slide_array, patch_size=patch_shape, step=step_size,
                                                                                       outdir=cache_dir, save_tiles=False, cores=cores)

                #measure time performance
                start_time = time.time()
                #pass raw slide_array for getting the initial concentrations, bg_reject_array for actual normalisation
                if norm:
                    logging.info(f"Normalising {slide_name}...")
                    canny_img, img_norm_wsi_jpg, canny_norm_patch_list, coords_list = normalizer.transform(slide_array, bg_reject_array, 
                                                                                                           rejected_tile_array, patch_shapes, cores=cores)
                    print(f"\n--- Normalised slide {slide_name}: {(time.time() - start_time)} seconds ---")
                    img_norm_wsi_jpg.save(slide_jpg) #save WSI.svs -> WSI.jpg

                else:
                    canny_img, canny_norm_patch_list, coords_list = get_raw_tile_list(slide_array.shape, bg_reject_array,
                                                                                      rejected_tile_array, patch_shapes)

                print("Saving Canny background rejected image...")
                canny_img.save(f'{slide_cache_dir}/canny_slide.jpg')
                
                #remove original slide jpg from memory
                del slide_array
                
                #optionally removing the original slide from harddrive
                if del_slide:
                    print(f"Deleting slide {slide_name} from local folder...")
                    os.remove(str(slide_url))

            print(f"Extracting CTransPath features from {slide_name}")
            #FEATURE EXTRACTION
            #measure time performance
            start_time = time.time()
            if len(canny_norm_patch_list) > 0:
                extract_features_(model=model, model_name=model_name, norm_wsi_img=canny_norm_patch_list,
                                coords=coords_list, wsi_name=slide_name, outdir=feat_out_dir, cores=cores, is_norm=norm, device=device if has_gpu else "cpu")
                print("\n--- Extracted features from slide: %s seconds ---" % (time.time() - start_time))
            else:
                print("0 tiles remain to extract features from after pre-processing {slide_name}, skipping...")
                continue
            #########################

        else:
            print(f"{slide_name}.h5 already exists. Skipping...")
            if del_slide:
                print(f"Deleting slide {slide_name} from local folder...")
                os.remove(str(slide_url))

    print(f"--- End-to-end processing time of {len(img_dir)} slides: {str(timedelta(seconds=(time.time() - total_start_time)))} ---")

def main():
    parser = argparse.ArgumentParser(
        description='Normalise WSI directly.')

    parser.add_argument('-o', '--output-dir', type=Path, required=True,
                        help='Path to save features to.')
    parser.add_argument('--wsi-dir', metavar='DIR', type=Path, required=True,
                        help='Path of where the whole-slide images are.')
    parser.add_argument('-m', '--model', metavar='DIR', type=Path, required=True,
                        help='Path of where model for the feature extractor is.')
    parser.add_argument('--cache-dir', type=Path, required=True, default=None,
        help='Directory to store resulting slide JPGs.')
    
    parser.add_argument('--patch-size', type=int, default=224,
                        help='Size of the square patch to tessellate.')
    parser.add_argument('--mpp', type=float, default=256/224,
                    help='Microns-per-pixel value for slide resolution.')
    parser.add_argument('-c', '--cores', type=int, default=8,
                    help='CPU cores to use, 8 default.')
    parser.add_argument('-n','--norm', action='store_true')
    parser.add_argument('--no-norm', dest='norm', action='store_false')
    parser.set_defaults(norm=True)
    parser.add_argument('-d', '--del-slide', action='store_true', default=False,
                         help='Removing the original slide after processing.')
    parser.add_argument('--only-fex', action='store_true', default=False)

    args = parser.parse_args()
    preprocess(output_dir=args.output_dir, wsi_dir=args.wsi_dir, model_path=args.model, cache_dir=args.cache_dir, patch_size=args.patch_size, target_mpp=args.mpp, cores=args.cores, norm=args.norm, del_slide=args.del_slide, only_feature_extraction=args.only_fex)


if __name__ == '__main__':
    main()