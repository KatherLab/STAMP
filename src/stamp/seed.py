import random
from typing import ClassVar

import numpy as np
import torch
from beartype.typing import Callable
from torch import Generator


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Seed:
    seed: ClassVar[int | None] = None

    @classmethod
    def torch(cls, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    @classmethod
    def python(cls, seed: int) -> None:
        random.seed(seed)

    @classmethod
    def numpy(cls, seed: int) -> None:
        np.random.seed(seed)

    @classmethod
    def set(cls, seed: int) -> None:
        cls.torch(seed)
        cls.python(seed)
        cls.numpy(seed)
        cls.seed = seed

    @classmethod
    def _is_set(cls) -> bool:
        return cls.seed is not None

    @classmethod
    def get_loader_worker_init(cls) -> Callable[[int], None]:
        if cls._is_set():
            return _seed_worker
        else:
            raise RuntimeError(
                "Seed has not been set. Call Seed.set(seed) before using a DataLoader."
            )

    @classmethod
    def get_torch_generator(cls, device="cpu") -> Generator:
        seed = cls.seed
        if seed is None:
            raise RuntimeError(
                "Seed has not been set. Call Seed.set(seed) before requesting a generator."
            )
        g = torch.Generator(device)
        g.manual_seed(seed)
        return g
