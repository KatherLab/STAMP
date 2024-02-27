import json
from pathlib import Path
from typing import Tuple
import hashlib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms #no marugoto dependency
from PIL import Image
import numpy as np
import h5py
from tqdm import tqdm

from .swin_transformer import swin_tiny_patch4_window7_224, ConvStem

__version__ = "001_01-10-2023"


class FeatureExtractor:
    """Extracts features from slide tiles."""
    def __init__(self, model: str, model_name: str, device : str = "cpu",):
        self.model_name = model_name
        self.model_type = "CTransPath"
        self.name = f'STAMP-extract-{__version__}_{model_name}'

        self.model = model
        self.model = self.model.to(device)
        self.model.eval()

        self.device = torch.device(device)
        self.dtype = next(self.model.parameters()).dtype

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str, device: str):
        # loading the checkpoint weights
        sha256 = hashlib.sha256()
        with open(checkpoint_path, 'rb') as f:
            while True:
                data = f.read(1 << 16)
                if not data:
                    break
                sha256.update(data)
        assert sha256.hexdigest() == '7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539'
        model_name='xiyuewang-ctranspath-7c998680'
        ctranspath_weights = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # initializing the model and updating the weights
        model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        model.head = nn.Identity()
        model.load_state_dict(ctranspath_weights['model'], strict=True)

        extractor = cls(model, model_name, device)
        print("CTransPath model successfully initialized...\n")

        return extractor

    def extract(self, patches: np.ndarray, cores: int = 8, batch_size: int = 64):
        """Extracts features from slide tiles.

        Args:
            slide_tile_paths:  A list of paths containing the slide tiles, one
                per slide.
            outdir:  Path to save the features to.
            augmented_repetitions:  How many additional iterations over the
                dataset with augmentation should be performed.  0 means that
                only one, non-augmentation iteration will be done.
        """
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        dataset = SlideTileDataset(patches, transform)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=cores,
            drop_last=False, pin_memory=self.device != torch.device('cpu')
        )

        features = []
        with torch.inference_mode():
            for patches_batch in tqdm(dataloader, leave=False):
                patches_batch = patches_batch.to(dtype=self.dtype, device=self.device)
                features_batch = self.model(patches_batch).half().cpu()
                features.append(features_batch)

        features = torch.concat(features, dim=0).numpy()
        return features


def store_metadata(outdir: Path, extractor_name: str,  patch_size: Tuple[int, int], target_microns: int, normalized: bool):
    with open(outdir.parent/'info.json', 'w') as f:
        json.dump(
            {
                'extractor': extractor_name,
                'augmented_repetitions': 0,
                'patches_normalized': normalized,
                'microns': target_microns,
                'patch_size': patch_size
            }, f
        )


def store_features(outdir: Path, features: np.ndarray, patches_coords: np.ndarray, extractor_name: str):
    with h5py.File(f'{outdir}.h5', 'w') as f:
        f['coords'] = patches_coords[:, ::-1] # store as (w, h) not (h, w) for backwards compatibility
        f['feats'] = features
        f['augmented'] = np.repeat(
            [False, True], [features.shape[0], 0])
        assert len(f['feats']) == len(f['augmented'])
        f.attrs['extractor'] = extractor_name


class SlideTileDataset(Dataset):
    def __init__(self, patches: np.array, transform=None, *, repetitions: int = 1) -> None:
        self.tiles = patches
        self.tiles *= repetitions
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        image = Image.fromarray(self.tiles[i])
        if self.transform:
            image = self.transform(image)

        return image
