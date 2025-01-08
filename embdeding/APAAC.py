#!/usr/bin/env python
#_*_coding:utf-8_*_
import re
import sys
import os
import math
import pandas as pd
import h5py
import numpy as np
from tqdm import tqdm


def APAAC(peptide_sequences, mhcii_sequences, lambdaValue=30, w=0.05):

    if min([len(seq) for seq in peptide_sequences]) < lambdaValue + 1 or min([len(seq) for seq in mhcii_sequences]) < lambdaValue + 1:
        raise ValueError(f"Error: All sequences must have length larger than lambdaValue + 1 = {lambdaValue + 1}")

    dataFile = "/home/data/PAAC.txt"
    
    with open(dataFile) as f:
        records = f.readlines()

    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {AA[i]: i for i in range(len(AA))}

    AAProperty = np.array([list(map(float, records[i].rstrip().split()[1:])) for i in range(1, len(records) - 1)])
    AAProperty1 = (AAProperty - AAProperty.mean(axis=1, keepdims=True)) / AAProperty.std(axis=1, keepdims=True)

    def encode_sequence(sequence):

        sequence = re.sub('-', '', sequence).replace('X', '')  
        code = []
        theta = []

        for n in range(1, lambdaValue + 1):
            for j in range(len(AAProperty1)):
                theta.append(np.mean([AAProperty1[j][AADict[sequence[k]]] * AAProperty1[j][AADict[sequence[k + n]]]
                                      for k in range(len(sequence) - n)]))

        myDict = {aa: sequence.count(aa) for aa in AA}
        code.extend([myDict[aa] / (1 + w * sum(theta)) for aa in AA])
        code.extend([w * t / (1 + w * sum(theta)) for t in theta])
        return code

    peptide_encodings = []
    mhcii_encodings = []

    for peptide_sequence, mhcii_sequence in tqdm(zip(peptide_sequences, mhcii_sequences), total=len(peptide_sequences), desc="Encoding sequences"):
        peptide_encodings.append(encode_sequence(peptide_sequence))
        mhcii_encodings.append(encode_sequence(mhcii_sequence))

    return np.array(peptide_encodings), np.array(mhcii_encodings) 



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

        h5f.attrs['description'] = 'BLOSUM62 encoded peptide and MHCII sequences with corresponding IC50 labels'

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

lambdaValue = 11

peptide_encodings, mhcii_encodings = APAAC(peptide_sequences, mhcii_sequences, lambdaValue=lambdaValue)

output_h5_path = "/home/data/APAAC.h5"
save_encoded_data_to_h5(peptide_encodings, mhcii_encodings, labels, output_h5_path)