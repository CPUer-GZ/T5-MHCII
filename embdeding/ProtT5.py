# -*- coding: utf-8 -*-
import os
import torch
from transformers import T5EncoderModel, T5Tokenizer
import time
import numpy as np
import pandas as pd
import re
import gc
import h5py
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Using device: {}".format(device))
########################################################################
def get_t5_model():
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
    model = model.to(device)
    model = model.eval()
    tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_uniref50', do_lower_case=False,legacy=True)
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

    dataset = pd.DataFrame({
        'peptide': peptides,
        'mhcii': mhcii_sequences,
        'label': ic50
    })

    return dataset


def get_embeddings(model, tokenizer, dataset):

    peptide_embs = [] 
    mhcii_embs = []   
    labels = []       
    total_sequences = len(dataset)

    for idx in tqdm(range(total_sequences), desc="Processing sequences"):
        peptide_seq = dataset['peptide'].iloc[idx]
        mhcii_seq = dataset['mhcii'].iloc[idx]
        current_label = dataset['label'].iloc[idx]

        processed_peptide_seq = " ".join(list(re.sub(r"[OBUZ]", "X", peptide_seq)))
        processed_mhcii_seq = " ".join(list(re.sub(r"[OBUZ]", "X", mhcii_seq)))

        peptide_token_encoding = tokenizer.batch_encode_plus(
            [processed_peptide_seq],
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )

        mhcii_token_encoding = tokenizer.batch_encode_plus(
            [processed_mhcii_seq],
            add_special_tokens=True,
            padding=True,
            return_tensors="pt"
        )

        peptide_input_ids = peptide_token_encoding['input_ids'].to(device)
        peptide_attention_mask = peptide_token_encoding['attention_mask'].to(device)

        mhcii_input_ids = mhcii_token_encoding['input_ids'].to(device)
        mhcii_attention_mask = mhcii_token_encoding['attention_mask'].to(device)

        with torch.no_grad():
            peptide_embedding = model(input_ids=peptide_input_ids, attention_mask=peptide_attention_mask)
            mhcii_embedding = model(input_ids=mhcii_input_ids, attention_mask=mhcii_attention_mask)

        peptide_embedding = peptide_embedding.last_hidden_state.cpu().numpy()
        mhcii_embedding = mhcii_embedding.last_hidden_state.cpu().numpy()

        peptide_len = (peptide_attention_mask.cpu().numpy() == 1).sum()
        mhcii_len = (mhcii_attention_mask.cpu().numpy() == 1).sum()

        peptide_seq_emb = peptide_embedding[0][:peptide_len - 1].mean(axis=0)
        mhcii_seq_emb = mhcii_embedding[0][:mhcii_len - 1].mean(axis=0)

        peptide_embs.append(peptide_seq_emb)
        mhcii_embs.append(mhcii_seq_emb)
        labels.append(current_label)

        torch.cuda.empty_cache()
        gc.collect()

    print(f"Total number of embeddings stored: Peptides: {len(peptide_embs)}, MHCII: {len(mhcii_embs)}")
    return np.array(peptide_embs), np.array(mhcii_embs), np.array(labels)

#########################################################################################################
def save_encoded_data_to_h5(peptide_encodings, mhcii_encodings, labels, output_h5_path):

    if peptide_encodings.shape[0] != mhcii_encodings.shape[0] or peptide_encodings.shape[0] != len(labels):
        raise ValueError("Peptide encodings, MHCII encodings, and labels must have the same number of samples.")

    with h5py.File(output_h5_path, 'w') as h5f:
       
        h5f.create_dataset('peptide_encodings', data=peptide_encodings, compression="gzip")
        h5f.create_dataset('mhcii_encodings', data=mhcii_encodings, compression="gzip")
        h5f.create_dataset('labels', data=labels, compression="gzip")

        h5f.attrs['description'] = 'ProstT5 encoded peptide and MHCII sequences with corresponding IC50 labels'

        h5f.attrs['peptide_encoding_shape'] = peptide_encodings.shape
        h5f.attrs['mhcii_encoding_shape'] = mhcii_encodings.shape
        h5f.attrs['num_samples'] = peptide_encodings.shape[0]
        h5f.attrs['num_features_peptide'] = peptide_encodings.shape[1]
        h5f.attrs['num_features_mhcii'] = mhcii_encodings.shape[1]

    print(f"Data successfully saved to {output_h5_path}")
#########################################################################################################
model, tokenizer = get_t5_model()

dataset = read_data("/home/data/PEPTIDE_MHCII_BA.csv", "/home/data/pseudosequence.2023.dat")

peptide_embs, mhcii_embs, labels = get_embeddings(model, tokenizer, dataset)

output_h5_path = "/home/data/representation_data/ProtT5.h5"
save_encoded_data_to_h5(peptide_embs, mhcii_embs, labels, output_h5_path)










