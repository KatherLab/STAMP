import hashlib
from pathlib import Path
import torch
import torch.nn as nn
import PIL
import numpy as np
#no marugoto dependency
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm
import json
import h5py

from .swin_transformer import swin_tiny_patch4_window7_224, ConvStem

__version__ = "001_01-10-2023"

class FeatureExtractor:
    def __init__(self):
        self.model_type = "CTransPath"

    def init_feat_extractor(self, checkpoint_path: str, device: str, **kwargs):
        """Extracts features from slide tiles.
        Args:
            checkpoint_path:  Path to the model checkpoint file.
        """
        sha256 = hashlib.sha256()
        with open(checkpoint_path, 'rb') as f:
            while True:
                data = f.read(1 << 16)
                if not data:
                    break
                sha256.update(data)

        assert sha256.hexdigest() == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'

        model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        model.head = nn.Identity()

        ctranspath = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(ctranspath['model'], strict=True)
        
        if torch.cuda.is_available():
            model = model.to(device)

        print("CTransPath model successfully initialised...\n")
        model_name='xiyuewang-ctranspath-7c998680'

        return model, model_name
        


class SlideTileDataset(Dataset):
    def __init__(self, patches: np.array, transform=None, *, repetitions: int = 1) -> None:
        self.tiles = patches
        #assert self.tiles, f'no tiles found in {slide_dir}'
        self.tiles *= repetitions
        self.transform = transform

    # patchify returns a NumPy array with shape (n_rows, n_cols, 1, H, W, N), if image is N-channels.
    # H W N is Height Width N-channels of the extracted patch
    # n_rows is the number of patches for each column and n_cols is the number of patches for each row
    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = PIL.Image.fromarray(self.tiles[i])
        if self.transform:
            image = self.transform(image)

        return image

def extract_features_(
        *,
        model, model_name, norm_wsi_img: np.ndarray, coords: list, wsi_name: str, outdir: Path,
        augmented_repetitions: int = 0, cores: int = 8, is_norm: bool = True, device: str = 'cpu',
        target_microns: int = 256, patch_size: int = 224
) -> None:
    """Extracts features from slide tiles.

    Args:
        slide_tile_paths:  A list of paths containing the slide tiles, one
            per slide.
        outdir:  Path to save the features to.
        augmented_repetitions:  How many additional iterations over the
            dataset with augmentation should be performed.  0 means that
            only one, non-augmentation iteration will be done.
    """

    normal_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    augmenting_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomVerticalFlip(p=.5),
        transforms.RandomApply([transforms.GaussianBlur(3)], p=.5),
        transforms.RandomApply([transforms.ColorJitter(
            brightness=.1, contrast=.2, saturation=.25, hue=.125)], p=.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    extractor_string = f'STAMP-extract-{__version__}_{model_name}'
    with open(outdir.parent/'info.json', 'w') as f:
        json.dump({'extractor': extractor_string,
                  'augmented_repetitions': augmented_repetitions,
                  'normalized': is_norm,
                  'microns': target_microns,
                  'patch_size': patch_size}, f)

    unaugmented_ds = SlideTileDataset(norm_wsi_img, normal_transform)
    augmented_ds = []

    #clean up memory
    del norm_wsi_img

    ds = ConcatDataset([unaugmented_ds, augmented_ds])
    dl = torch.utils.data.DataLoader(
        ds, batch_size=64, shuffle=False, num_workers=cores, drop_last=False, pin_memory=device != 'cpu')

    model = model.eval().to(device)
    dtype = next(model.parameters()).dtype

    feats = []
    for batch in tqdm(dl, leave=False):
        feats.append(
            model(batch.type(dtype).to(device)).half().cpu().detach())

    with h5py.File(f'{outdir}.h5', 'w') as f:
        f['coords'] = coords
        f['feats'] = torch.concat(feats).cpu().numpy()
        f['augmented'] = np.repeat(
            [False, True], [len(unaugmented_ds), len(augmented_ds)])
        assert len(f['feats']) == len(f['augmented'])
        f.attrs['extractor'] = extractor_string