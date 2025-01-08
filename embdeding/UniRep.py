# -*- coding: utf-8 -*-
import torch
from jax_unirep import get_reps
import numpy as np
import re
import pandas as pd
import h5py
from tqdm import tqdm
import gc
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

########################################################################
def read_data(csv_file_path, MHCIIdata_file_path):

    try:
        df = pd.read_csv(csv_file_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file: {e}")

    required_columns = ['Peptide', 'Alleles', 'IC50']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"CSV文件必须包含列: {required_columns}")

    peptides = df['Peptide']
    alleles = df['Alleles']
    ic50 = df['IC50']

    alleles_to_mhcii = {}
    try:
        with open(MHCIIdata_file_path, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    allele, mhcii_sequence = parts
                    alleles_to_mhcii[allele] = mhcii_sequence
    except Exception as e:
        raise ValueError(f"Error reading data file: {e}")

    mhcii_sequences = alleles.map(alleles_to_mhcii)

    missing_alleles = mhcii_sequences.isnull().sum()
    if missing_alleles > 0:
        missing_alleles_list = alleles[mhcii_sequences.isnull()].tolist()
        print(f"Warning: {missing_alleles} alleles have no corresponding MHCII sequence. Filling with 'unknown'.")
        print(f"Missing alleles: {missing_alleles_list}")
        mhcii_sequences = mhcii_sequences.fillna('unknown')

    for idx, (peptide, mhcii, label) in enumerate(zip(peptides, mhcii_sequences, ic50)):
        if pd.isnull(peptide) or pd.isnull(mhcii) or pd.isnull(label):
            raise ValueError(f"Sample at index {idx} is incomplete: Peptide={peptide}, MHCII={mhcii}, IC50={label}")

    dataset = pd.DataFrame({
        'peptide': peptides,
        'mhcii': mhcii_sequences,
        'label': ic50
    })

    return dataset


from tqdm import tqdm
import numpy as np
import gc

def get_embeddings(dataset, batch_size=1024):
    peptide_embs = []  
    mhcii_embs = []  
    labels = []       
    total_sequences = len(dataset)

    for start_idx in tqdm(range(0, total_sequences, batch_size), desc="Processing sequences"):
        end_idx = min(start_idx + batch_size, total_sequences)

        peptide_seqs = dataset['peptide'].iloc[start_idx:end_idx].tolist()
        mhcii_seqs = dataset['mhcii'].iloc[start_idx:end_idx].tolist()
        current_labels = dataset['label'].iloc[start_idx:end_idx].tolist()

        peptide_h_avg, _, _ = get_reps(peptide_seqs)
        mhcii_h_avg, _, _ = get_reps(mhcii_seqs)

        peptide_embs.append(peptide_h_avg)
        mhcii_embs.append(mhcii_h_avg)
        labels.extend(current_labels)

        gc.collect()
        torch.cuda.empty_cache()

    peptide_embs = np.vstack(peptide_embs)
    mhcii_embs = np.vstack(mhcii_embs)

    print(f"Total number of peptide embeddings stored: {peptide_embs.shape[0]}")
    print(f"Total number of mhcii embeddings stored: {mhcii_embs.shape[0]}")

    return peptide_embs, mhcii_embs, np.array(labels)

def save_encoded_data_to_h5(peptide_encodings, mhcii_encodings, labels, output_h5_path):

    if peptide_encodings.shape[0] != mhcii_encodings.shape[0] or peptide_encodings.shape[0] != len(labels):
        raise ValueError("Peptide encodings, MHCII encodings, and labels must have the same number of samples.")

    with h5py.File(output_h5_path, 'w') as h5f:
        h5f.create_dataset('peptide_encodings', data=peptide_encodings, compression="gzip")
        h5f.create_dataset('mhcii_encodings', data=mhcii_encodings, compression="gzip")
        h5f.create_dataset('labels', data=labels, compression="gzip")

        h5f.attrs['description'] = 'UniRep encoded peptide and MHCII sequences with corresponding IC50 labels'

        h5f.attrs['peptide_encoding_shape'] = peptide_encodings.shape
        h5f.attrs['mhcii_encoding_shape'] = mhcii_encodings.shape
        h5f.attrs['num_samples'] = peptide_encodings.shape[0]
        h5f.attrs['num_features_peptide'] = peptide_encodings.shape[1]
        h5f.attrs['num_features_mhcii'] = mhcii_encodings.shape[1]

    print(f"Data successfully saved to {output_h5_path}")


dataset = read_data("/home/data/PEPTIDE_MHCII_BA.csv", "/home/data/pseudosequence.2023.dat")

peptide_embs, mhcii_embs, labels = get_embeddings(dataset)

output_h5_path =  "/home/data/representation_data/UniRep.h5"
save_encoded_data_to_h5(peptide_embs, mhcii_embs, labels, output_h5_path)