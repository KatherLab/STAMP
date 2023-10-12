# %%
#from ast import Return
from typing import Tuple
from PIL import Image
from matplotlib import pyplot as plt
from pydantic import NoneBytes
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn as nn
from pathlib import Path
import pandas as pd


# %%
__all__ = ['plot_GCAM_from_csv_top_tiles_']


def _get_GCAM_map(img, model: Tuple[nn.Module],
                  idx_target_class: int) -> torch.Tensor:
    """
    Gives heat map for layer between model_to_fmap and model_after_fmap \
         parts of the model

    Args:
        img: Image to calculate GradCam for
        model: Tuple of len 2, contains:
           model[0]: model_to_fmap: part of the model up to 
           layer of interest
           model[1]: model_after_fmap: part of model beyond
           model ResNet in MIL example:
            body = nn.Sequential(*list(someResNet_base.children())[:-2])
            head = nn.Sequential(AdaptiveConcatPool2d(), nn.Flatten(),
                MIL_head.encoder,MIL_head.head)
            model = (body,head)
        idx_target_class: index of target class to calculate GradCam for

    Returns:
        GradCam heatmap  
    """

    assert len(model) == 2, f'model should be a tuple of len 2, \
        size given {len(model)}. '
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_to_fmap, model_after_fmap = model
    model_to_fmap = model_to_fmap.half().eval().to(device)
    model_after_fmap = model_after_fmap.half().eval().to(device)

    normal_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    x_in = normal_transform(img).half().to(device)

    # add bs dimension
    if x_in.ndim == 3:
        x_in = x_in.unsqueeze(0)

    feature_maps = model_to_fmap(x_in)
    feature_maps.retain_grad()
    act_out = model_after_fmap(feature_maps)
    act_out = act_out[0, idx_target_class]
    act_out.backward()

    # dim hmap = n_bs, n_feat, x_, y_,
    hmap = feature_maps*feature_maps.grad
    # remove bs dimension
    hmap = hmap.squeeze()
    # sum over number of features to get a x_*y_ sized map
    hmap = hmap.sum(0)
    # only get positive contributions as stated in grad_cam paper
    hmap = nn.ReLU()(hmap)

    return hmap.cpu().detach()


def _plot_GCAM_map_from_file_on_axes(image_path: Path, model: Tuple[nn.Module],
                                     idx_target_class: int, ax_in: plt.axes) -> plt.axes:

    img = Image.open(image_path)
    x = img.size[0]
    y = img.size[1]
    GCAM_map = _get_GCAM_map(
        img, model, idx_target_class=idx_target_class)
    ax_in.imshow(img)
    # y x or x y??
    ax_in.imshow(GCAM_map, alpha=0.5, extent=(0, y, x, 0),
                 interpolation='bilinear', cmap='magma')
    ax_in.axis('off')
    return ax_in


def plot_GCAM_from_csv_top_tiles_(
        out_dir: Path, t_tiles_csv_path: Path, model: nn.Module, idx_target_class: int = 0,
        n_tiles: int = 25, n_tiles_per_row: int = 5, outfile: Path=None):

    if not out_dir:
        out_dir = Path(t_tiles_csv_path).parent

    df_t_tiles = pd.read_csv(t_tiles_csv_path)
    n_in = min(n_tiles, df_t_tiles.shape[0])
    n_rows = n_in//n_tiles_per_row
    df_in = df_t_tiles.iloc[:n_in, :]
    fig, axs = plt.subplots(n_rows, n_tiles_per_row, figsize=(10, 10), dpi=300)

    for idx, ax_in in enumerate(axs.reshape(-1)):
        fname = df_in['file_name'][idx]
        ax_in = _plot_GCAM_map_from_file_on_axes(
            fname, model, idx_target_class, ax_in)
    if outfile:
        out=out_dir/outfile
    else:
        out = out_dir/'GCAM_on_top_tiles.jpg'
    fig.savefig(out)
    print(f'Saved figure to {out}.')
    return

