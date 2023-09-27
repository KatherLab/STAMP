# %%
from enum import Enum, auto
from typing import Mapping, Optional, Sequence, Tuple
from fastai.vision.learner import load_learner
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import torch.nn as nn
from marugoto.mil.data import get_target_enc
from matplotlib.patches import Patch
from scipy import interpolate
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import h5py

__all__ = ['plot_heatmaps_', 'MapType']
# list of allowed formats for whole slide images
wsi_suffixes = ['.svs', '.ndpi', '.tif']
# define colours for heatmap plots here
colors = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 1], [1, 1, 0]])


class MapType(Enum):
    ATTENTION = auto()
    PROBABILITY = auto()
    CONTRIBUTION = auto()


def get_dict_maptype_to_coords_scores(h5_feature_path: Path, model: nn.Module,
                                      map_types: list[MapType] = [
                                          MapType.ATTENTION],
                                      ) -> dict:

    dict_maptype_to_coords_scores = {}

    feats, coords, sizes = [], [], []
    with h5py.File(h5_feature_path, 'r') as f:
        feats.append(torch.from_numpy(f['feats'][:]).float())
        sizes.append(len(f['feats']))
        coords.append(torch.from_numpy(f['coords'][:]))
    feats, coords = torch.cat(feats), torch.cat(coords)

    encoder = model.encoder.eval()
    attention = model.attention.eval()
    head = model.head.eval()

    # calculate attention, scores etc.
    encs = encoder(feats)
    patient_atts = torch.softmax(attention(encs), dim=0).detach()
    patient_scores = torch.softmax(head(encs), dim=1).detach()
    normed_patient_atts=(patient_atts-patient_atts.min())/(patient_atts.max()-patient_atts.min())
    patient_weighted_scores=normed_patient_atts*patient_scores

    assert patient_scores.shape[-1] <= colors.shape[0], f'not enough colours.\n'\
        'Can only plot score for max {colors.shape[0]}'\
        'classes at a time!\n Number of classes asked for:'\
        f'{len()} not supported.'

    for map_type in map_types:
        match map_type:
            case MapType.ATTENTION:
                scores = patient_atts.numpy()
                scores -= scores.min()
                scores /= (scores.max()-scores.min())
            case MapType.PROBABILITY:
                scores = patient_scores.numpy()
            case MapType.CONTRIBUTION:
                scores = patient_weighted_scores.numpy()
            case _:
                raise ValueError(f'heat map type {map_type} not supported!')

        dict_maptype_to_coords_scores[map_type] = coords.numpy(), scores

    return dict_maptype_to_coords_scores


def _MIL_heatmap_for_slide(coords: np.ndarray, scores: np.ndarray,
                           colours: np.ndarray = None) -> np.ndarray:
    """
    Args: 
        h5_feature_path: path to .h5 file with features to analyse
        model: model to analyse slide with
        categories: TODO
        map_type: one from ['attention','probability', contribution]

    Returns:
        Tuple of covered_area, legend, heatmap
        covered_area: extent in x dimension and extent in y dimension of whole slide image
        legend: details of legend for plot
        heatmap: the actual heatmap, z coordinate is activation, x and y are integers from 0 to n_x
            and 0 to n_y, respectively. To get pixel dimensions multiply x and y by stride
    """

    # get stride
    stride = _get_stride(coords)
    scaled_map_coords = coords // stride
    if colours is not None:
        pass
    else:
        colours = colors

    # make a mask, 1 where coordinates have attention 0 otherwise
    # ndarray of zeros of dimension max_x * max_y
    mask = np.zeros(scaled_map_coords.max(0) + 1)
    # add in ones where we have values
    for coord in scaled_map_coords:
        mask[coord[0], coord[1]] = 1

    grid_x, grid_y = np.mgrid[0:scaled_map_coords[:, 0].max()+1,
                              0:scaled_map_coords[:, 1].max()+1]

    # interpolate heatmap over grid
    if scores.ndim < 2:
        scores = np.expand_dims(scores, 1)
    activations = interpolate.griddata(
        scaled_map_coords, scores, (grid_x, grid_y))
    activations = np.nan_to_num(activations) * np.expand_dims(mask, 2)

    heatmap = _visualize_activation_map(
        activations.transpose(1, 0, 2), colours[:activations.shape[-1]])

    return heatmap


def _plot_heatmap_(coords, heatmap, legend_elements,
                   outdir: Path, wsi_path: Optional[Path] = None,
                   superimpose: bool = True, alpha: int = 0.5,
                   ) -> None:

    format = '.svg'
    plt.figure(dpi=600)

    stride = _get_stride(coords)
    covered_area = (coords.max(0)+stride)

    if wsi_path:
        from openslide import OpenSlide

        assert wsi_path.suffix in wsi_suffixes, \
            f'cannot read files with extension {wsi_path.suffix}. ' \
            f'Please provide a WSI with extension in {wsi_suffixes}.'

        slide = OpenSlide(str(wsi_path))

        # get the first level smaller than max_size
        # FIXME: replace with get_thumbnail?
        level = next((i for i, dims in enumerate(slide.level_dimensions)
                     if max(dims) <= 2400*2),
                     slide.level_count-1)
        thumb = slide.read_region(
            (0, 0), level, slide.level_dimensions[level])
        covered_area_size = (
            covered_area / slide.level_downsamples[level]).astype(int)

        if superimpose:
            # make heatmap transparent by putting alpha_channel values <1!
            heatmap[:, :, -1] = heatmap[:, :, -1]*alpha
        heatmap = Image.fromarray(heatmap)
        # make heatmap and thumb the same size
        scaled_heatmap = Image.new('RGBA', thumb.size)
        scaled_heatmap.paste(heatmap.resize(
            covered_area_size, resample=Image.Resampling.NEAREST))
        if superimpose:
            thumb.alpha_composite(scaled_heatmap)
            plt.imshow(thumb)
            plt.axis('off')
        else:
            _, axs = plt.subplots(1, 2, figsize=(12, 6), dpi=300)
            axs[0].imshow(thumb)
            axs[0].axis('off')
            axs[1].imshow(scaled_heatmap)
            axs[1].axis('off')
    else:
        print(f'No path to WSI given, plotting heatmap without WSI ...\n')
        # only plot heatmap
        heatmap = Image.fromarray(heatmap)
        heatmap = heatmap.resize(np.multiply(
            heatmap.size, 8), resample=Image.Resampling.NEAREST)
        plt.imshow(heatmap)
        plt.axis('off')

    title = wsi_path.stem

    legend = plt.legend(
        title=title, handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')

    out_file = (outdir/wsi_path.stem).with_suffix(format)
    print(f'Writing output to file: {out_file}')
    out_file.parent.mkdir(exist_ok=True, parents=True)

    plt.savefig(out_file, bbox_extra_artists=[legend], bbox_inches='tight')
    plt.close('all')


def _get_stride(coordinates: np.ndarray) -> int:
    xs = sorted(set(coordinates[:, 0]))
    x_strides = np.subtract(xs[1:], xs[:-1])

    ys = sorted(set(coordinates[:, 1]))
    y_strides = np.subtract(ys[1:], ys[:-1])

    stride = min(*x_strides, *y_strides)

    return stride


def _visualize_activation_map(activations: np.ndarray, colors: np.ndarray, alpha: float = 1.,
    clipping: bool=True) -> np.ndarray:
    """Transforms an activation map into an RGBA numpy array.
    Args:
        activations: An (h, w, D) array of activations.
        colors: A (D, 3) array mapping each of the target classes to a color.
    Returns:
        An interpolated activation map. Regions which the algorithm assumes to be background
        will be transparent.
    """
    assert colors.shape[1] == 3, "expected color map to have three color channels"
    assert colors.shape[0] == activations.shape[2], "one color map entry per class required"
    # activations should be less or equal to 1
    assert activations[2].max() <= 1, f"Activations should be less than one, otherwise maps get clipped! \n \
        Max value provided {activations[2].max()}."
    # transform activation map into RGB map
    rgbmap = activations.dot(colors)
    # TODO this is a cheap fix only!
    if clipping:
        max_cvalue=np.amax(rgbmap)
        if max_cvalue>255.0:
            #Rescale
            rgbmap=rgbmap/max_cvalue*255.0
            print(f"Rescaled rgbmap as max pixel value is {max_cvalue}")
            print(f"This could potentially be a problem!")

    # create RGBA map with non-zero activations being the foreground
    mask = activations.any(axis=2)

    # mask * alpha gives alpha at non zero values of activation
    # below gives value for the alpha channel
    im_data = (np.concatenate([rgbmap, np.expand_dims(
        mask * alpha, -1)], axis=2) * 255.5).astype(np.uint8)

    return im_data


def _get_slide_features(h5_feature_dir, ws_path):

    assert h5_feature_dir.is_dir(), \
        f'{h5_feature_dir} is not a directory. Please provide path to feature directory!'

    whole_slides = []
    h5_feature_paths = []

    if ws_path.is_file():
        whole_slides.append(ws_path)
        h5_feature_path = h5_feature_dir/ws_path.with_suffix('.h5').name
        h5_feature_paths.append(h5_feature_path)
    elif ws_path.is_dir():
        for suffix in wsi_suffixes:
            for ws_p in ws_path.glob(f'*{suffix}'):
                h5_feature_path = h5_feature_dir/ws_p.with_suffix('.h5').name
                if h5_feature_path.is_file():
                    whole_slides.append(ws_p)
                    h5_feature_paths.append(h5_feature_path)
                else:
                    print(
                        f'Could not find file {h5_feature_path}.\
                             Check if features where extracted for this whole slide image!')
    else:
        raise ValueError(
            f'Given ws_path is neither a file nor a directory. Path given {ws_path=!r}.')

    return list(zip(whole_slides, h5_feature_paths))


def plot_heatmaps_(out_dir: Path, train_dir: Path, ws_path: Path, h5_feature_dir: Path,
                  map_types: list[MapType] = [MapType.ATTENTION],
                  superimpose: bool = True, alpha: float = 0.5):
    """Generates heatmaps for whole slide images.

    Outputs heatmaps to project directory, in subfolders for each map_type.

    Args:
        out_dir: path to where outputs are stored
        train_dir: path to directory where training was done, i.e. where export.pkl file is located
        ws_path: path to whole slide image, either full path -> single image is analysed or directory 
            -> all whole slide images in directory are analysed
        h5_feature_dir: directory containing features used in training, must match whole slide images
        map_types: list containing attention, probability and/or contribution to give corresponding heatmaps
        superimpose: have heatmap on top of thumbnail or both side-by-side
        alpha: transparacy of heatmap
    """
    slide_features = _get_slide_features(h5_feature_dir, ws_path)

    learn = load_learner(train_dir/'export.pkl')
    target_enc = get_target_enc(learn)
    categories = target_enc.categories_[0]

    str_targets = ['contrib_'+target for target in categories]

    for slide_path, h5_feature_path in slide_features:
        dict_maptype_to_coords_scores = get_dict_maptype_to_coords_scores(h5_feature_path,
                                                                          model=learn.model, map_types=map_types)
        for map_type in map_types:
            coords, scores = dict_maptype_to_coords_scores[map_type]
            if map_type == MapType.ATTENTION:
                legend_elements = [
                    Patch(facecolor=colors[0], label='attention')]
            else:
                legend_elements = [Patch(facecolor=color, label=class_) for class_,
                                   color in zip(str_targets, colors)]
            heatmap = _MIL_heatmap_for_slide(coords=coords, scores=scores)

            _plot_heatmap_(coords, heatmap=heatmap, legend_elements=legend_elements,
                           wsi_path=slide_path, superimpose=superimpose, outdir=out_dir/map_type.name,
                           alpha=alpha)


def get_overlay(thumb,covered_area_size, coords, scores, alpha=0.6, colors=colors):
    """ takes a thumb image, resizes it to covered_area_size, gets heatmap for scores
        and overlays score heatmap over thumb image.
    """
# get attention map in overlay
    heatmap = _MIL_heatmap_for_slide(coords=coords, scores= scores,
                            colours=colors)
    heatmap[:, :, -1] = heatmap[:, :, -1]*alpha
    heatmap = Image.fromarray(heatmap)
    # make heatmap and thumb the same size
    scaled_heatmap = Image.new('RGBA', thumb.size)
    scaled_heatmap.paste(heatmap.resize(
        covered_area_size, resample=Image.Resampling.NEAREST))
    return Image.alpha_composite(thumb, scaled_heatmap)

# below are just two examples of how one can use above functions to plot
# your own heatmaps

def plot_heatmaps_two_cats_(out_dir: Path, train_dir1: Path, train_dir2: Path, ws_path: Path, h5_feature_dir: Path,
                              superimpose: bool = True, alpha: float = 0.6):
    
    format = '.svg'
    plt.figure(dpi=600)

    slide_features = _get_slide_features(h5_feature_dir, ws_path)

    learn1 = load_learner(train_dir1/'export.pkl')
    target_enc1 = get_target_enc(learn1)
    categories1 = target_enc1.categories_[0]

    learn2 = load_learner(train_dir2/'export.pkl')
    target_enc2 = get_target_enc(learn1)
    categories2 = target_enc2.categories_[0]

    from openslide import OpenSlide
    map_types = [MapType.ATTENTION, MapType.PROBABILITY]

    for slide_path, h5_feature_path in slide_features:
        dict_maptype_to_coords_scores1 = get_dict_maptype_to_coords_scores(
            h5_feature_path, model=learn1.model, map_types=map_types)
        coords, att_scores1 = dict_maptype_to_coords_scores1[MapType.ATTENTION]
        dict_maptype_to_coords_scores2 = get_dict_maptype_to_coords_scores(
            h5_feature_path, model=learn2.model, map_types=map_types)
        
        _, att_scores2 = dict_maptype_to_coords_scores2[MapType.ATTENTION]

        stride = _get_stride(coords)
        covered_area = (coords.max(0)+stride)

        # get thumbnail

        assert slide_path.suffix in wsi_suffixes, \
            f'cannot read files with extension {slide_path.suffix}. ' \
            f'Please provide a WSI with extension in {wsi_suffixes}.'

        slide = OpenSlide(str(slide_path))
        level = next((i for i, dims in enumerate(slide.level_dimensions)
                     if max(dims) <= 2400*2),
                     slide.level_count-1)
        thumb = slide.read_region(
            (0, 0), level, slide.level_dimensions[level])
        covered_area_size = (
            covered_area / slide.level_downsamples[level]).astype(int)

        # get attention map in overlay
        att_overlay1 = get_overlay(thumb,covered_area_size,coords,att_scores1,alpha=alpha)
        att_overlay2 = get_overlay(thumb,covered_area_size,coords,att_scores2,alpha=alpha)
        _, probs1 = dict_maptype_to_coords_scores1[MapType.PROBABILITY]
        _,probs2 = dict_maptype_to_coords_scores2[MapType.PROBABILITY]
        prob_overlay1 = get_overlay(thumb,covered_area_size,coords,probs1[:,0],alpha=alpha)
        color = np.expand_dims(colors[1, :], axis=0)
        prob_overlay2 = get_overlay(thumb,covered_area_size,coords,probs2[:,0],alpha=alpha,
            colors=color)
        prob_tot=np.stack([probs1[:,0],probs2[:,0]],axis=1)
        prob_tot_over=get_overlay(thumb,covered_area_size,coords,prob_tot,alpha=alpha)
        _, axs = plt.subplots(2, 3, figsize=(12, 6), dpi=300)
        axs[0, 0].imshow(thumb)
        axs[0, 0].legend(title='thumb')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(att_overlay1)
        axs[0, 1].axis('off')
        axs[0, 1].legend(title='attention')
        axs[0, 2].imshow(att_overlay2)
        axs[0, 2].axis('off')
        axs[1,0].imshow(prob_overlay1)
        legend_elements = [Patch(facecolor=colors[0], label=f'prob_{categories1[0]}')]
        axs[1, 0].legend(title='scores', handles=legend_elements,
                         bbox_to_anchor=(1, 1), loc='upper left')
        axs[1, 0].axis('off')
        axs[1,1].imshow(prob_overlay2)
        legend_elements = [Patch(facecolor=colors[1], label=f'prob_{categories2[0]}')]
        axs[1, 1].legend(title='scores', handles=legend_elements,
                         bbox_to_anchor=(1, 1), loc='upper left')
        axs[1, 1].axis('off')

        axs[1,2].imshow(prob_overlay2)
        axs[1, 2].imshow(prob_tot_over)
        legend_elements = [Patch(facecolor=color, label=class_) for class_,
                           color in zip(['MSIH','braf'], [colors[0],colors[1]])]
        axs[1, 2].legend(title='contribution', handles=legend_elements,
                         bbox_to_anchor=(1, 1), loc='upper left')
        axs[1, 2].axis('off')

        out_file = (out_dir/f'CRC_paper_{slide_path.stem}').with_suffix(format)
        print(f'Writing output to file: {out_file}')
        out_file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_file, bbox_inches='tight')
        plt.close('all')

    return

def plot_heatmaps_CRC_RAINBOW_(out_dir: Path, train_dir: Path, ws_path: Path, h5_feature_dir: Path,
                               alpha: float = 0.6):

    format = '.svg'
    plt.figure(dpi=600)

    slide_features = _get_slide_features(h5_feature_dir, ws_path)

    learn = load_learner(train_dir/'export.pkl')
    target_enc = get_target_enc(learn)
    categories = target_enc.categories_[0]

    str_targets = ['contrib_'+target for target in categories]
    categories = categories.tolist()
    while len(categories) < 4:
        categories.append(None)

    from openslide import OpenSlide
    map_types = [MapType.ATTENTION, MapType.CONTRIBUTION]
    for slide_path, h5_feature_path in slide_features:
        dict_maptype_to_coords_scores = get_dict_maptype_to_coords_scores(
            h5_feature_path, model=learn.model, map_types=map_types)
        coords, att_scores = dict_maptype_to_coords_scores[MapType.ATTENTION]

        stride = _get_stride(coords)
        covered_area = (coords.max(0)+stride)

        # get thumbnail

        assert slide_path.suffix in wsi_suffixes, \
            f'cannot read files with extension {slide_path.suffix}. ' \
            f'Please provide a WSI with extension in {wsi_suffixes}.'

        slide = OpenSlide(str(slide_path))
        level = next((i for i, dims in enumerate(slide.level_dimensions)
                     if max(dims) <= 2400*2),
                     slide.level_count-1)
        thumb = slide.read_region(
            (0, 0), level, slide.level_dimensions[level])
        covered_area_size = (
            covered_area / slide.level_downsamples[level]).astype(int)

        # get attention map in overlay
        heatmap = _MIL_heatmap_for_slide(coords=coords, scores=att_scores)
        heatmap[:, :, -1] = heatmap[:, :, -1]*alpha
        heatmap = Image.fromarray(heatmap)
        # make heatmap and thumb the same size
        scaled_heatmap = Image.new('RGBA', thumb.size)
        scaled_heatmap.paste(heatmap.resize(
            covered_area_size, resample=Image.Resampling.NEAREST))
        att_overlay = Image.alpha_composite(thumb, scaled_heatmap)

        _, cont_scores = dict_maptype_to_coords_scores[MapType.CONTRIBUTION]
        heatmap = _MIL_heatmap_for_slide(coords=coords, scores=cont_scores)
        heatmap[:, :, -1] = heatmap[:, :, -1]*alpha
        heatmap = Image.fromarray(heatmap)
        # make heatmap and thumb the same size
        scaled_heatmap = Image.new('RGBA', thumb.size)
        scaled_heatmap.paste(heatmap.resize(
            covered_area_size, resample=Image.Resampling.NEAREST))
        cont_overlay = Image.alpha_composite(thumb, scaled_heatmap)

        _, axs = plt.subplots(2, 4, figsize=(12, 6), dpi=300)
        axs[0, 0].imshow(thumb)
        axs[0, 0].legend(title='thumb')
        axs[0, 0].axis('off')
        axs[0, 1].imshow(att_overlay)
        axs[0, 1].axis('off')
        axs[0, 1].legend(title='attention')
        axs[0, 2].imshow(cont_overlay)
        axs[0, 2].axis('off')
        legend_elements = [Patch(facecolor=color, label=class_) for class_,
                           color in zip(str_targets, colors)]
        axs[0, 2].legend(title='contribution', handles=legend_elements,
                         bbox_to_anchor=(1, 1), loc='upper left')
        axs[0, 3].axis('off')

        for i, cat in enumerate(categories):
            if cat is not None:
                color = np.expand_dims(colors[i, :], axis=0)
                heatmap0 = _MIL_heatmap_for_slide(
                    coords=coords, scores=cont_scores[:, i], colours=color)
                heatmap0 = Image.fromarray(heatmap0)
                scaled_heatmap0 = Image.new('RGBA', thumb.size)
                legend_elements = [Patch(facecolor=colors[i], label=cat)]
                scaled_heatmap0.paste(heatmap0.resize(
                    covered_area_size, resample=Image.Resampling.NEAREST))
                axs[1, i].legend
                axs[1, i].imshow(scaled_heatmap0)
                axs[1, i].axis('off')
                axs[1, i].legend(handles=legend_elements)
            else:
                axs[1, i].axis('off')

        out_file = (out_dir/f'CRC_paper_{slide_path.stem}').with_suffix(format)
        print(f'Writing output to file: {out_file}')
        out_file.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(out_file, bbox_inches='tight')
        plt.close('all')