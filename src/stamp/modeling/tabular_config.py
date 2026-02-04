import numpy as np
import pandas as pd
import torch

TAB_COLS = ["Age", "sexe", "plaq0", "albu0", "afp0"]


def encode_tabular(df: pd.DataFrame):
    # validate schema
    missing = set(TAB_COLS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing tabular columns: {missing}")

    df = df.copy()

    # categorical
    x_categ = (
        df["sexe"]
        .astype(str)
        .str.lower()
        .map({"f": 0, "Féminin": 0, "m": 1, "Masculin": 1})
        .astype(int)
        .values
    )
    x_categ = torch.tensor(x_categ, dtype=torch.long).unsqueeze(1)

    # numerical
    df["Age"] = df["Age"].astype(float)
    df["plaq0"] = df["plaq0"].astype(float)
    df["albu0"] = df["albu0"].astype(float)
    df["afp0"] = np.log1p(df["afp0"].astype(float))

    x_numer = torch.tensor(
        df[["Age", "plaq0", "albu0", "afp0"]].values,
        dtype=torch.float32,
    )

    return x_categ, x_numer
