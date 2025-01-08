import torch
import h5py
from torch.utils.data import Dataset

class BindDataset(Dataset):
    def __init__(self, h5_file_path):
        self.h5_file_path = h5_file_path

        with h5py.File(h5_file_path, 'r') as h5_file:
            self.peptide_encodings = h5_file['pep_hidden_states'][:]
            self.mhcii_encodings = h5_file['mhcii_hidden_states'][:]
            self.combined_encodings = h5_file['combined_hidden_states'][:]
            self.peptide_masks = h5_file['peptide_masks'][:]
            self.mhcii_masks = h5_file['mhcii_masks'][:]
            self.combined_masks = h5_file['combined_masks'][:]
            self.labels = h5_file['labels'][:]
            self.alleles = [allele.decode('utf-8') for allele in h5_file['alleles'][:]]
        
        self.length = self.peptide_encodings.shape[0]

    def __len__(self):
        return self.length  

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError("Index out of range")

        return {
            'peptide_encoding': torch.tensor(self.peptide_encodings[idx], dtype=torch.float32),
            'mhcii_encoding': torch.tensor(self.mhcii_encodings[idx], dtype=torch.float32),
            'combined_encoding': torch.tensor(self.combined_encodings[idx], dtype=torch.float32),
            'peptide_mask': torch.tensor(self.peptide_masks[idx], dtype=torch.float32),
            'mhcii_mask': torch.tensor(self.mhcii_masks[idx], dtype=torch.float32),
            'combined_mask': torch.tensor(self.combined_masks[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32),
            'alleles': self.alleles[idx]
        }

