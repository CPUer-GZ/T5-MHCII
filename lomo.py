import os
import re
import torch
import pandas as pd
import random
import numpy as np
import math
from torch.utils.data import Subset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split,KFold,LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr,ttest_rel
import copy
from tqdm import tqdm
from model import T5ForPepMHCIIRegression
from data_processing import BindDataset
import h5py

def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self,
                 dataset,
                 epochs = 10,
                 lr = 0.001,
                 seed = 42
                ):
        self.dataset = dataset
        self.epochs = epochs
        self.lr = lr
        set_seed(seed)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.loss_fn = torch.nn.MSELoss()
        torch.manual_seed(seed)
        
    def train(self):
        self.model.to(self.device)
        self.model.train()

        total_loss = 0
        for i, batch in enumerate(tqdm(self.train_loader, desc='Training Batch', leave=False, dynamic_ncols=True)):
            peptide_hidden_states = batch['peptide_encoding'].to(self.device)
            peptide_mask = batch['peptide_mask'].to(self.device)

            mhcii_hidden_states = batch['mhcii_encoding'].to(self.device)
            mhcii_mask = batch['mhcii_mask'].to(self.device)

            combined_hidden_states = batch['combined_encoding'].to(self.device)
            combined_mask = batch['combined_mask'].to(self.device)

            labels = batch['labels'].to(self.device)

            logits = self.model(
                peptide_hidden_states=peptide_hidden_states,
                peptide_mask=peptide_mask,
                mhcii_hidden_states=mhcii_hidden_states,
                mhcii_mask=mhcii_mask,
                combined_hidden_states=combined_hidden_states,
                combined_mask=combined_mask,
            )
            loss = self.loss_fn(logits.view(-1), labels.view(-1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
   
            self.optimizer.step()
 
            self.optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss


    def validate(self):
        self.model.eval()

        total_loss = 0
        predictions = []
        labels = []
        alleles = []

        with torch.no_grad():
            for batch in tqdm(self.valid_loader, desc='Validation Batch', leave=False, dynamic_ncols=True):
                peptide_hidden_states = batch['peptide_encoding'].to(self.device)
                peptide_mask = batch['peptide_mask'].to(self.device)
                mhcii_hidden_states = batch['mhcii_encoding'].to(self.device)
                mhcii_mask = batch['mhcii_mask'].to(self.device)
                combined_hidden_states = batch['combined_encoding'].to(self.device)
                combined_mask = batch['combined_mask'].to(self.device)
                labels_batch = batch['labels'].to(self.device)
                alleles_batch = batch['alleles']

                logits = self.model(
                    peptide_hidden_states=peptide_hidden_states,
                    peptide_mask=peptide_mask,
                    mhcii_hidden_states=mhcii_hidden_states,
                    mhcii_mask=mhcii_mask,
                    combined_hidden_states=combined_hidden_states,
                    combined_mask=combined_mask,
                )

                loss = self.loss_fn(logits.view(-1), labels_batch.view(-1))

                total_loss += loss.item()
                predictions.append(logits)
                labels.append(labels_batch)
                alleles.extend(alleles_batch)

        val_labels = torch.cat(labels).cpu().numpy()
        val_predictions = torch.cat(predictions).cpu().numpy()

        threshold = 1 - (math.log(500) / math.log(50000))
        binary_labels = (val_labels >= threshold).astype(int)

        pcc, _ = spearmanr(val_predictions, val_labels)
        auc = roc_auc_score(binary_labels, val_predictions)

        return total_loss / len(self.valid_loader), auc, pcc, val_predictions, val_labels, alleles

    def lomo(self):
        print("***** Running LOMO *****")
        
        mhc_name = [
                    # 'DRB1_0101', 
                    # 'DRB1_0103', 
                    # 'DRB1_1301', 
                    # 'DRB1_0701',
                    # 'H-2-IAu', 
                    # 'H-2-IEk',
                    # 'H-2-IAb',
                    # 'HLA-DPA10103-DPB10402', 
                    # 'HLA-DPA10103-DPB10301',
                    # 'HLA-DQA10101-DQB10501', 
                    # 'HLA-DQA10501-DQB10201', 
                    # 'HLA-DQA10501-DQB10302',
                    'HLA-DPA10201-DPB10101',
                   ]
    
        alleles = [self.dataset[i]['alleles'] for i in range(len(self.dataset))]
        
        valid_splits = [(train_idx, test_idx) for train_idx, test_idx in LeaveOneGroupOut().split(range(len(self.dataset)), groups=alleles) if alleles[test_idx[0]] in mhc_name]
        
        output_dir = os.path.expanduser('~/autodl-tmp/lomo_results')
        os.makedirs(output_dir, exist_ok=True)
        
        best_allele_metrics = {}

        for fold, (train_idx, test_idx) in enumerate(tqdm(valid_splits, desc='Validation Loop', leave=False, dynamic_ncols=True)):
            test_allele = alleles[test_idx[0]]
            print(f"Validation Fold {fold + 1}/{len(valid_splits)} - Test Allele: {test_allele}")

            train_subset = Subset(self.dataset, train_idx)
            valid_subset = Subset(self.dataset, test_idx)
    
            self.train_loader = DataLoader(train_subset, batch_size=512, shuffle=True, num_workers=12)
            self.valid_loader = DataLoader(valid_subset, batch_size=512, shuffle=False, num_workers=12)
    
            self.model = T5ForPepMHCIIRegression()
            self.model.to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.001)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1, min_lr=0.000001)
    
            best_auc = 0
            best_pcc = 0
            best_epoch = 0
    
            for epoch in tqdm(range(self.epochs), desc='Epoch Loop', leave=False, dynamic_ncols=True):
                train_loss = self.train()
                val_loss, val_auc, val_pcc, val_predictions, val_labels, alleles = self.validate()
    
                current_lr = self.scheduler.optimizer.param_groups[0]['lr']
    
                tqdm.write(f'Epoch {epoch + 1}\n'
                           f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}\n'
                           f'Val AUC: {val_auc:.4f} | Val Pearsonr: {val_pcc:.4f}\n'
                           f'LR: {current_lr:.6f}\n'
                           '****************************************')
                
                self.scheduler.step(val_loss)
    
                if val_auc > best_auc:
                    best_auc = val_auc
                    best_pcc = val_pcc
                    best_epoch = epoch

                    best_allele_metrics[test_allele] = {'auc': best_auc, 'pcc': best_pcc}

                torch.cuda.empty_cache()
    
            print(f"Best AUC for {test_allele} - Fold {fold + 1}: {best_auc:.4f}, Best PCC: {best_pcc:.4f} at Epoch {best_epoch + 1}")

            del self.model
            del self.optimizer
            del self.scheduler
            torch.cuda.empty_cache()
    
        best_allele_metrics_path = os.path.join(output_dir, 'best_allele_metrics.csv')
        best_allele_metrics_df = pd.DataFrame.from_dict(best_allele_metrics, orient='index').reset_index()
        best_allele_metrics_df.columns = ['Allele', 'AUC', 'PCC']
        best_allele_metrics_df.to_csv(best_allele_metrics_path, index=False)
        print(f"Best per-allele metrics saved to {best_allele_metrics_path}")


if __name__ == "__main__":

    h5_file_path = os.path.expanduser('~/encoded_data.h5') # Path to the encoded data file

    dataset = BindDataset(h5_file_path)

    trainer = Trainer(
        dataset=dataset,
        epochs=10,
        lr=0.001,
        seed=315,
    )

    trainer.lomo()
