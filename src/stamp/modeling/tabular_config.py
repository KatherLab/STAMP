from typing import TextIO

import numpy as np
import pandas as pd
import torch

TAB_COLS = ["Age", "sexe", "plaq0", "albu0", "afp0"]


def encode_tabular(df: pd.DataFrame):
    df = df.copy()

    # ---------------------------
    # categorical: sexe
    # ---------------------------
    sex_encoded = (
        df["sexe"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "féminin": 0,
                "masculin": 1,
            }
        )
    )

    if sex_encoded.isna().any():
        bad = df.loc[sex_encoded.isna(), "sexe"].unique()
        raise ValueError(f"Unrecognized sexe values: {bad}")

    x_categ = torch.tensor(sex_encoded.values, dtype=torch.long).unsqueeze(1)

    # ---------------------------
    # numerical: normalize decimals + cast
    # ---------------------------
    NUM_COLS = ["Age", "plaq0", "albu0", "afp0"]

    for col in NUM_COLS:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", ".", regex=False)  # decimal commas
            .astype(float)
        )

    # log-transform AFP after numeric conversion
    # df["afp0"] = np.log1p(df["afp0"])
    # df["plaq0"] = np.log1p(df["plaq0"])
    # df["albu0"] = np.log1p(df["albu0"])
    # df["Age"] = np.log1p(df["Age"])
    # df["Age"] = df["Age"].astype(float)
    # df["Age"] = df["Age"] - df["Age"].median()
    # df["afp0"] = df["afp0"].rank(pct=True)
    # df["plaq0"] = np.sqrt(df["plaq0"].clip(lower=0))
    # df["albu0"] = df["albu0"].astype(float)

    # for col in ["Age", "afp0", "plaq0", "albu0"]:
    #     df[f"{col}_missing"] = df[col].isna().astype(int)

    #     x = df[col].fillna(df[col].median())

    #     df[f"{col}_rank"] = x.rank(pct=True)

    #     # optional: keep raw but robust-scaled
    #     q1, q3 = x.quantile([0.25, 0.75])
    #     df[col] = (x - q1) / (q3 - q1)

    # ---------------------------
    # hard safety check (IMPORTANT)
    # ---------------------------
    if df[NUM_COLS].isna().any().any():
        bad_rows = df[df[NUM_COLS].isna().any(axis=1)]
        raise ValueError(
            "NaNs detected in tabular numerical features after encoding.\n"
            f"Offending rows (first 5):\n{bad_rows[NUM_COLS].head()}"
        )

    x_numer = torch.tensor(
        df[NUM_COLS].values,
        dtype=torch.float32,
    )

    return x_categ, x_numer

# from pathlib import Path


# def read_table(path: Path | TextIO, **kwargs) -> pd.DataFrame:
#     if not isinstance(path, Path):
#         return pd.read_csv(path, **kwargs)

#     if path.suffix == ".xlsx":
#         return pd.read_excel(path, **kwargs)

#     if path.suffix == ".csv":
#         try:
#             # First try UTF-8 (modern, correct)
#             return pd.read_csv(path, **kwargs)
#         except UnicodeDecodeError:
#             # Fallback for legacy clinical data (Excel / SPSS / Windows)
#             return pd.read_csv(path, encoding="latin-1", **kwargs)

#     raise ValueError(
#         "table to load has to either be an excel (`*.xlsx`) or csv (`*.csv`) file."
#     )


# csv_path = "/mnt/copernicus3/PATHOLOGY/others/private/GENIAL/metadata/GENIAL_ANAPATH_CirVir_CIRRAL/edited/cirral_cirvir_time_to_hcc.xlsx"  # <-- path here
# # ---- load table (this tests encoding handling) ----
# df = read_table(Path(csv_path))

# print("Loaded table shape:", df.shape)
# print("Columns:", df.columns.tolist())

# # ---- select only tabular columns ----
# df_tab = df[TAB_COLS]

# # ---- encode ----
# x_categ, x_numer = encode_tabular(df_tab)

# # ---- assertions ----
# assert isinstance(x_categ, torch.Tensor)
# assert isinstance(x_numer, torch.Tensor)

# assert x_categ.ndim == 2 and x_categ.shape[1] == 1
# assert x_numer.ndim == 2 and x_numer.shape[1] == 4

# assert x_categ.dtype == torch.long
# assert x_numer.dtype == torch.float32

# # ---- sanity checks ----
# print("x_categ shape:", x_categ.shape)
# print("x_numer shape:", x_numer.shape)
# print("Unique sexe codes:", torch.unique(x_categ).tolist())
# print("Numerical means:", x_numer.mean(dim=0))
# print("Numerical stds:", x_numer.std(dim=0))

# print("✅ encode_tabular test passed")

# # ---- survival sanity check (RAW TABLE) ----
# assert "HCC_E" in df.columns, "HCC_E column missing"
# assert "time_to_HCC" in df.columns, "time_to_HCC column missing"

# # normalize (defensive)
# df["HCC_E"] = pd.to_numeric(df["HCC_E"], errors="coerce")
# df["time_to_HCC"] = df["time_to_HCC"].astype(str).str.replace(",", ".", regex=False)
# df["time_to_HCC"] = pd.to_numeric(df["time_to_HCC"], errors="coerce")

# # drop invalid
# df_surv = df.dropna(subset=["HCC_E", "time_to_HCC"])

# print("RAW survival samples:", len(df_surv))
# print("RAW event count:", int(df_surv["HCC_E"].sum()))
# print("RAW event rate:", df_surv["HCC_E"].mean())

# assert df_surv["HCC_E"].sum() > 0, "❌ No events in raw table after cleaning"