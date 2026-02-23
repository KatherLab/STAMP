import torch
from jaxtyping import Float


def vary_precision(
    data: Float[torch.Tensor, "*dims"], *, min_fraction_bits: int
) -> Float[torch.Tensor, "*dims"]:
    """Randomly reduces the precision of the tensor's values."""
    if min_fraction_bits < 1:
        raise ValueError("min_fraction bits has to be at least 1")

    if data.dtype == torch.float32:
        fraction_bits = 23
        mask_dtype = torch.int32
    elif data.dtype == torch.float16:
        fraction_bits = 10
        mask_dtype = torch.int16
    elif data.dtype == torch.bfloat16:
        fraction_bits = 7
        mask_dtype = torch.int16
    else:
        raise NotImplementedError(
            f"precision variation not implemented for {data.dtype}"
        )

    no_of_bits_to_mask = torch.randint(0, fraction_bits - min_fraction_bits, data.shape)
    mask = (~0 << no_of_bits_to_mask).to(dtype=mask_dtype, device=data.device)
    augmented = (data.view(mask_dtype) & mask).view(data.dtype)
    return augmented


class VaryPrecisionTransform:
    """A transform randomly reducing the precision of its inputs."""

    def __init__(self, *, min_fraction_bits: int = 1) -> None:
        self.min_fraction_bits = min_fraction_bits

    def __call__(
        self, batch: Float[torch.Tensor, "*dims"]
    ) -> Float[torch.Tensor, "*dims"]:
        return vary_precision(data=batch, min_fraction_bits=self.min_fraction_bits)
