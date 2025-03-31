from transformers import AutoModel

from stamp.slide_encoding.encoder import Encoder


def titan() -> Encoder:
    model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    # TODO: Check precision

    return Encoder(
        model=model,
        identifier="mahmood-titan",
    )
