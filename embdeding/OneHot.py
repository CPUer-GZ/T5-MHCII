# -*- coding: utf-8 -*-
import torch
from bio_embeddings.embed import OneHotEncodingEmbedder
import numpy as np
import re
import pandas as pd
import h5py
from tqdm import tqdm
device = torch.device('cpu')

print("Using device: {}".format(device))
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


def get_embeddings(dataset):

    peptide_embs = []
    mhcii_embs = []
    labels = []
    total_sequences = len(dataset)

    embedder = OneHotEncodingEmbedder()

    for idx in tqdm(range(total_sequences), desc="Processing sequences"):
        peptide_seq = dataset['peptide'].iloc[idx]
        mhcii_seq = dataset['mhcii'].iloc[idx]
        current_label = dataset['label'].iloc[idx]

        with torch.no_grad():
            peptide_embeds = embedder.embed(peptide_seq)
            peptide_emb_per_protein = np.mean(peptide_embeds, axis=0) 

        with torch.no_grad():
            mhcii_embeds = embedder.embed(mhcii_seq)
            mhcii_emb_per_protein = np.mean(mhcii_embeds, axis=0)  

        peptide_embs.append(peptide_emb_per_protein)
        mhcii_embs.append(mhcii_emb_per_protein)
        labels.append(current_label)

    print(f"Total number of peptide embeddings stored: {len(peptide_embs)}")
    print(f"Total number of mhcii embeddings stored: {len(mhcii_embs)}")
    return np.array(peptide_embs), np.array(mhcii_embs), np.array(labels)


def save_encoded_data_to_h5(peptide_encodings, mhcii_encodings, labels, output_h5_path):

    if peptide_encodings.shape[0] != mhcii_encodings.shape[0] or peptide_encodings.shape[0] != len(labels):
        raise ValueError("Peptide encodings, MHCII encodings, and labels must have the same number of samples.")


    with h5py.File(output_h5_path, 'w') as h5f:

        h5f.create_dataset('peptide_encodings', data=peptide_encodings, compression="gzip")
        h5f.create_dataset('mhcii_encodings', data=mhcii_encodings, compression="gzip")
        h5f.create_dataset('labels', data=labels, compression="gzip")

        h5f.attrs['description'] = 'BLOSUM62 encoded peptide and MHCII sequences with corresponding IC50 labels'

        h5f.attrs['peptide_encoding_shape'] = peptide_encodings.shape
        h5f.attrs['mhcii_encoding_shape'] = mhcii_encodings.shape
        h5f.attrs['num_samples'] = peptide_encodings.shape[0]
        h5f.attrs['num_features_peptide'] = peptide_encodings.shape[1]
        h5f.attrs['num_features_mhcii'] = mhcii_encodings.shape[1]

    print(f"Data successfully saved to {output_h5_path}")

dataset = read_data("/home/data/PEPTIDE_MHCII_BA.csv", "/home/data/pseudosequence.2023.dat")

peptide_embs, mhcii_embs, labels = get_embeddings(dataset)


output_h5_path = "/home/data/representation_data/OneHot.h5"
save_encoded_data_to_h5(peptide_embs, mhcii_embs, labels, output_h5_path)