# -*- coding: utf-8 -*-
import torch
from tape import ProteinBertModel, TAPETokenizer
import numpy as np
import re
import pandas as pd
import h5py
from tqdm import tqdm

import os
os.environ['TORCH_HOME'] = r'D:\\OneDrive\\我要毕业\\model_location'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))

########################################################################
def get_tape_model():
    model = ProteinBertModel.from_pretrained('bert-base')
    model = model.to(device)
    model = model.eval()
    tokenizer = TAPETokenizer(vocab='iupac')
    return model, tokenizer

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
        
    concatenated_sequences = peptides + mhcii_sequences

    dataset = pd.DataFrame({
        'concatenated': concatenated_sequences, 
        'label': ic50 
    })

    return dataset


def get_embeddings(model, tokenizer, seqs, label):

    protein_embs = []
    labels = []
    total_sequences = len(seqs)

    for seq_idx in tqdm(range(total_sequences), desc="Processing sequences"):
        current_seq = seqs[seq_idx]
        current_label = label[seq_idx]

        token_ids = torch.tensor([tokenizer.encode(current_seq)]).to(device)

        with torch.no_grad():
            embedding_repr = model(token_ids)[0]  

        emb_per_protein = embedding_repr.mean(axis=0)  

        protein_embs.append(emb_per_protein)
        labels.append(current_label)

    print(f"Total number of per-protein embeddings stored: {len(protein_embs)}")
    return protein_embs, labels

#########################################################################################################
def save_encoded_data_to_h5(encoded_sequences, labels, output_h5_path):

    encoded_sequences = np.array(encoded_sequences)
    labels = np.array(labels)

    print(f"Encoded sequences shape: {encoded_sequences.shape}")
    print(f"Labels shape: {labels.shape}")

    with h5py.File(output_h5_path, 'w') as h5f:
        embeddings_dataset = h5f.create_dataset("embeddings", shape=encoded_sequences.shape, dtype='float32')
        labels_dataset = h5f.create_dataset("labels", shape=labels.shape, dtype='float32')

        print("Saving encoded sequences and labels...")
        for i in tqdm(range(encoded_sequences.shape[0]), desc="Saving data"):
            embeddings_dataset[i] = encoded_sequences[i]
            labels_dataset[i] = labels[i]

        print(f"Saved {len(encoded_sequences)} encodings and {len(labels)} labels to {output_h5_path}.")
#########################################################################################################
model, tokenizer = get_tape_model()

dataset = read_data("/home/data/PEPTIDE_MHCII_BA.csv", "/home/data/pseudosequence.2023.dat")
seqs = dataset['concatenated'].tolist()
label = dataset['label'].values

protein_embs, labels = get_embeddings(model, tokenizer, seqs, label)

output_h5_path = "/home/data/representation_data/TAPE_BERT_PFAM.h5"
save_encoded_data_to_h5(protein_embs, labels, output_h5_path)