import os
from pathlib import Path

import gdown
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import ndarray
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm

from stamp.cache import STAMP_CACHE_DIR, file_digest
from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import PandasLabel
from stamp.preprocessing.config import ExtractorName

"""authors: https://github.com/hms-dbmi/CHIEF"""


class CHIEFModel(nn.Module):
    def __init__(
        self,
        gate=True,
        size_arg="large",
        dropout=True,
        n_classes=2,
        instance_loss_fn=nn.CrossEntropyLoss(),
        **kwargs,
    ):
        super(CHIEFModel, self).__init__()
        self.size_dict = {
            "xs": [384, 256, 256],
            "small": [768, 512, 256],
            "big": [1024, 512, 384],
            "large": [2048, 1024, 512],
        }
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        if dropout:
            fc.append(nn.Dropout(0.25))
        if gate:
            attention_net = Attn_Net_Gated(
                L=size[1], D=size[2], dropout=dropout, n_classes=1
            )
        else:
            attention_net = Attn_Net(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for i in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        initialize_weights(self)

        self.att_head = Att_Head(size[1], size[2])
        self.text_to_vision = nn.Sequential(
            nn.Linear(768, size[1]), nn.ReLU(), nn.Dropout(p=0.25)
        )

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.attention_net = self.attention_net.to(device)
        self.classifiers = self.classifiers.to(device)
        self.instance_classifiers = self.instance_classifiers.to(device)

    def forward(self, h):
        h_ori = h
        A, h = self.attention_net(h)
        A = torch.transpose(A, 1, 0)
        A_raw = A
        A = F.softmax(A, dim=1)
        WSI_feature = torch.mm(A, h)
        slide_embeddings = torch.mm(A, h_ori)

        result = {
            "attention_raw": A_raw,
            "WSI_feature": slide_embeddings,
            "WSI_feature_transformed": WSI_feature,
            "tile_features_transformed": h,
        }
        return result


class CHIEF(Encoder):
    def __init__(self) -> None:
        model = CHIEFModel(size_arg="small", dropout=True, n_classes=2)
        model_path = STAMP_CACHE_DIR / "CHIEF_pretraining.pth"
        if not model_path.is_file():
            model_path.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(
                "https://drive.google.com/u/0/uc?id=10bJq_ayX97_1w95omN8_mESrYAGIBAPb&export=download",
                str(model_path),
            )
            digest = file_digest(model_path)
            assert (
                digest
                == "6a46d200b32a65e5ce4774611b889b5f1bbf7a39f9111321a2a1b5dbdb9996b8"
            ), (
                f"The digest of the downloaded checkpoint ({model_path}) did not match the expected value."
            )

        chief = torch.load(model_path)
        if "organ_embedding" in chief:
            del chief["organ_embedding"]
        model.load_state_dict(chief, strict=True)
        super().__init__(
            model=model,
            identifier=EncoderName.CHIEF,
            precision=torch.float32,
            required_extractor=[ExtractorName.CHIEF_CTRANSPATH],
        )

    def _generate_slide_embedding(
        self, feats: torch.Tensor, device, **kwargs
    ) -> ndarray:
        slide_embedding = self.model(feats.to(device))["WSI_feature"]
        return slide_embedding.detach().squeeze().cpu().numpy()

    def _generate_patient_embedding(
        self, feats_list: list, device, **kwargs
    ) -> ndarray:
        all_feats = torch.cat(feats_list, dim=0).to(device)

        patient_embedding = self.model(all_feats.to(device))["WSI_feature"]
        return patient_embedding.detach().squeeze().cpu().numpy()

    def encode_patients(
        self,
        output_dir: Path,
        feat_dir: Path,
        slide_table_path: Path,
        patient_label: PandasLabel,
        filename_label: PandasLabel,
        device: DeviceLikeType,
        generate_hash: bool,
        **kwargs,
    ) -> None:
        slide_table = pd.read_csv(slide_table_path)
        patient_groups = slide_table.groupby(patient_label)

        output_file = self._generate_output_path(
            output_dir=output_dir, generate_hash=generate_hash
        )

        patient_dict = {}
        self.model.to(device).eval()

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        for patient_id, group in tqdm(patient_groups, leave=False):
            feats_list = []

            # Concatenate all slides over x axis adding the offset to each feature x coordinate.
            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                h5_path = os.path.join(feat_dir, slide_filename)

                try:
                    feats, _ = self._validate_and_read_features(h5_path=h5_path)
                except FileNotFoundError as e:
                    tqdm.write(s=str(e))
                    continue

                feats_list.append(feats)

            if not feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            all_feats = torch.cat(feats_list, dim=0).to(device)

            patient_embedding = (
                self.model(all_feats.to(device))["WSI_feature"]
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            patient_dict[patient_id] = {
                "feats": patient_embedding,
            }

        self._save_features(output_file=output_file, entry_dict=patient_dict)


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Att_Head(nn.Module):
    def __init__(self, FEATURE_DIM, ATT_IM_DIM):
        super(Att_Head, self).__init__()

        self.fc1 = nn.Linear(FEATURE_DIM, ATT_IM_DIM)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(ATT_IM_DIM, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x


class Attn_Net(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]

        if dropout:
            self.module.append(nn.Dropout(0.25))

        self.module.append(nn.Linear(D, n_classes))

        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # type: ignore # N x 1, N * D


class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]

        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout:
            self.attention_a.append(nn.Dropout(0.25))
            self.attention_b.append(nn.Dropout(0.25))

        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)

        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x):
        a = self.attention_a(x)  # type: ignore
        b = self.attention_b(x)  # type: ignore
        A = a.mul(b)
        A = self.attention_c(A)
        return A, x
