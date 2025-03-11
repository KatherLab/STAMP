try:
    from cobra.utils.load_cobra import get_cobra
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "cobra dependencies not installed."
        " Please update your venv using `uv sync --extra cobra`"
    ) from e
from stamp.slide_encoding.encoder import Encoder


def cobra() -> Encoder:
    model = get_cobra(download_weights=True)
    return Encoder(
        model=model,
        identifier="katherlab-cobra",
    )