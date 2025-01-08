#!/usr/bin/env python
#_*_coding:utf-8_*_

import re
from collections import Counter
import pandas as pd
import h5py
from Bio.SeqUtils.ProtParam import ProteinAnalysis 
import numpy as np
from tqdm import tqdm

def frequency(peptide_sequences, mhcii_sequences, **kw):

    def calculate_frequency(sequence):

        analyzed_seq = ProteinAnalysis(sequence)
        freq = analyzed_seq.get_amino_acids_percent()  
        return freq  

    amino_acids_standard = sorted(ProteinAnalysis('ACDEFGHIKLMNPQRSTVWY').get_amino_acids_percent().keys())

    def encode_sequence(sequence):

        freq_dict = calculate_frequency(sequence)
        
        return [freq_dict.get(aa, 0) for aa in amino_acids_standard]

    peptide_encodings = []
    for peptide in tqdm(peptide_sequences, desc="Encoding peptides"):
        peptide_encodings.append(encode_sequence(peptide))
    peptide_encodings = np.array(peptide_encodings)  

    mhcii_encodings = []
    for mhcii in tqdm(mhcii_sequences, desc="Encoding MHCII sequences"):
        mhcii_encodings.append(encode_sequence(mhcii))
    mhcii_encodings = np.array(mhcii_encodings) 

    return peptide_encodings, mhcii_encodings


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


def save_encoded_data_to_h5(peptide_encodings, mhcii_encodings, labels, output_h5_path):

    if peptide_encodings.shape[0] != mhcii_encodings.shape[0] or peptide_encodings.shape[0] != len(labels):
        raise ValueError("Peptide encodings, MHCII encodings, and labels must have the same number of samples.")

    with h5py.File(output_h5_path, 'w') as h5f:

        h5f.create_dataset('peptide_encodings', data=peptide_encodings, compression="gzip")
        h5f.create_dataset('mhcii_encodings', data=mhcii_encodings, compression="gzip")
        h5f.create_dataset('labels', data=labels, compression="gzip")

        h5f.attrs['description'] = 'frequency encoded peptide and MHCII sequences with corresponding IC50 labels'

        h5f.attrs['peptide_encoding_shape'] = peptide_encodings.shape
        h5f.attrs['mhcii_encoding_shape'] = mhcii_encodings.shape
        h5f.attrs['num_samples'] = peptide_encodings.shape[0]
        h5f.attrs['num_features_peptide'] = peptide_encodings.shape[1]
        h5f.attrs['num_features_mhcii'] = mhcii_encodings.shape[1]

    print(f"Data successfully saved to {output_h5_path}")



dataset = read_data("/home/data/PEPTIDE_MHCII_BA.csv", "/home/data/pseudosequence.2023.dat")

peptide_sequences = dataset['peptide'].values
mhcii_sequences = dataset['mhcii'].values
labels = dataset['label'].values

peptide_encodings, mhcii_encodings = frequency(peptide_sequences, mhcii_sequences)

output_h5_path = "/home/data/frequency.h5"
save_encoded_data_to_h5(peptide_encodings, mhcii_encodings, labels, output_h5_path)