import torch
from transformers import AutoModel

from stamp.encoding.encoder import Encoder


class Prism(Encoder):
    def __init__(self) -> None:
        model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
        precision = torch.float16
        required_extractor = "virchow_full"
        super().__init__(
            model=model,
            identifier="paigeai-prism",
            precision=precision,
            required_extractor=required_extractor,
        )

    def _generate_slide_embedding(self, feats, device, **kwargs):
        with torch.autocast(device, dtype=self.precision), torch.inference_mode():
            return (
                self.model.slide_representations(feats.unsqueeze(0).to(device))[
                    "image_embedding"
                ]
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )

    def _generate_patient_embedding(self, feats_list, device, **kwargs):
        all_feats = torch.cat(feats_list, dim=0).to(device)
        with torch.inference_mode():
            return (
                self.model.slide_representations(all_feats.unsqueeze(0))[
                    "image_embedding"
                ]
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
