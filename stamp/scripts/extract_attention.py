#!/usr/bin/env python3

__author__ = "Thomas Sorz-Nechay"
__version__ = "1.0.0"
__email__ = "thomas.sorz-nechay@meduniwien.ac.at"

import pandas as pd
import torch
import os
from pathlib import Path
from fastai.learner import load_learner
import argparse
import h5py
from tqdm import tqdm

'''
This script extracts RAW attention weights from attMIL models. 
NOTE: If cross-patient comparisons are made in any way, keep in mind the attention weights should be normalized first. 

Output: .csv file
'''

df_list = [] # List of dataframes containing att-weights for patient - concatenated in fin_df
attention_list = [] # Placeholder attention weights of current .h5 file

parser = argparse.ArgumentParser("Extracts tile-wise attention weights")

parser.add_argument("path_h5", type=str, help="Provide path to .h5 directory")
parser.add_argument("path_model", type=str, help="Provide path to the pickled model")
parser.add_argument("path_out", type=str, help="Provide output path")
parser.add_argument("name", type=str, help="Provide a name for the output file (without extension)")

args = parser.parse_args()

model_path = Path(args.path_model)
data_dir = Path(args.path_h5)
path_out = Path(args.path_out + args.name)
name = args.name

# Hook function
def hook_function(module, inputs, output):
    attention_weight = torch.softmax(output, dim=1).squeeze(-1)
    attention_list.append(attention_weight.detach().cpu().numpy()[0])

# Load the model
learn = load_learner(model_path)
model = learn.model
model.eval()

# Register the hook to the attention layer
hook_handle = model.attention.register_forward_hook(hook_function)

# Iterate over h5 files
n_files = len([f for f in os.scandir(data_dir) if f.name.endswith(".h5")])

for bag_file in tqdm(data_dir.glob("*.h5"), total=n_files):

    file_name = bag_file.stem # Get file name
    h5_file = h5py.File(bag_file, 'r') # Read .h5 file

    # Extract tile features, coordinates and num_tiles
    feats = torch.tensor(h5_file['feats'][:]).float() 
    tensor_coords = torch.tensor(h5_file['coords'][:]).detach().cpu().numpy()
    
    num_tiles = torch.tensor([len(feats)])

    # Perform inference
    with torch.no_grad():
        model(feats.unsqueeze(0), num_tiles)
    
    # list of coordinates formatted as "X-coord_Y-coord" string
    coords = ["-".join(map(str, row)) for row in tensor_coords]
    df_index = [f"{file_name}_{c}" for c in coords]

    to_df = {"Coords":coords, "Attention":attention_list[-1].copy()}
    attention_list.clear()
    
    df = pd.DataFrame(data=to_df, columns=["Coords", "Attention"])
    df.index = df_index

    df_list.append(df.copy())

# Remove the hook
hook_handle.remove()

# Save output
fin_df = pd.concat(df_list, axis=0)
fin_df.to_csv(f"{path_out}.csv")

print("DataFrame created and saved successfully.")