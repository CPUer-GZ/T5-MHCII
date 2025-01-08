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

from sklearn.model_selection import train_test_split,KFold
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

            # 前向传播
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

    def fivecv(self):
        print("***** Running 5-fold cross-validation *****")

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_results = []
        output_dir = os.path.expanduser('~/prott5_results')
        os.makedirs(output_dir, exist_ok=True)
    
        for fold, (train_idx, valid_idx) in enumerate(tqdm(kf.split(self.dataset), desc='Fold_loop', leave=False, dynamic_ncols=True)):
    
            train_subset = Subset(self.dataset, train_idx)
            valid_subset = Subset(self.dataset, valid_idx)
    
            self.train_loader = DataLoader(train_subset, batch_size=512, shuffle=True, num_workers=12)
            self.valid_loader = DataLoader(valid_subset, batch_size=512, shuffle=False, num_workers=12)
    
            self.model = T5ForPepMHCIIRegression()
            self.model.to(self.device)
            self.optimizer = AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.001)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=1, min_lr=0.000001)
            self.best_scores = {}  
            best_auc = 0
            best_pcc = 0
            best_epoch = 0
            fold_group_results = {}
            best_model_path = os.path.join(output_dir, f'best_model_{fold + 1}_fold.pth')
    
            best_allele_metrics = {}  
    
            for epoch in tqdm(range(self.epochs), desc='Epoch_loop', leave=False, dynamic_ncols=True):
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
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'auc': best_auc,
                        'pcc': best_pcc,
                        'loss': val_loss
                    }, best_model_path)
                    print(f"Best model saved with AUC: {best_auc} at epoch {epoch + 1}")

                    unique_alleles = sorted(set(alleles))
                    for allele in unique_alleles:
                        allele_mask = (np.array(alleles) == allele)
                        allele_labels = np.array(val_labels)[allele_mask]
                        allele_preds = np.array(val_predictions)[allele_mask]
                        if len(allele_labels) > 0:
                            threshold = 1 - (math.log(500) / math.log(50000))
                            binary_allele_labels = (allele_labels >= threshold).astype(int)
                            if len(np.unique(binary_allele_labels)) == 2:
                                try:
                                    allele_auc = roc_auc_score(binary_allele_labels, allele_preds)
                                    allele_pcc, _ = spearmanr(allele_preds, allele_labels)
                                    best_allele_metrics[allele] = {'auc': allele_auc, 'pcc': allele_pcc}
                                except ValueError as e:
                                    print(f"Error calculating AUC for allele {allele}: {e}")
    
                torch.cuda.empty_cache()
    
            print(f"Best AUC for Fold {fold + 1}: {best_auc:.4f}, Best PCC: {best_pcc:.4f} at Epoch {best_epoch + 1}")
    
            print(f"Group evaluation results for the best model in Fold {fold + 1}:")
            group_data = []
            for allele, metrics in best_allele_metrics.items():
                group_data.append([allele, metrics['auc'], metrics['pcc']])
            avg_group_auc = np.mean([metrics['auc'] for metrics in best_allele_metrics.values()])
            avg_group_pcc = np.mean([metrics['pcc'] for metrics in best_allele_metrics.values()])
            std_group_auc = np.std([metrics['auc'] for metrics in best_allele_metrics.values()])
            std_group_pcc = np.std([metrics['pcc'] for metrics in best_allele_metrics.values()])
            if len(best_allele_metrics) > 1:
                p_value_auc = ttest_rel(
                    [metrics['auc'] for metrics in best_allele_metrics.values()],
                    [avg_group_auc] * len(best_allele_metrics)
                )[1]
                p_value_pcc = ttest_rel(
                    [metrics['pcc'] for metrics in best_allele_metrics.values()],
                    [avg_group_pcc] * len(best_allele_metrics)
                )[1]
            else:
                p_value_auc = float('nan')
                p_value_pcc = float('nan')
            group_data.append(['AVG', avg_group_auc, avg_group_pcc])
            group_data.append(['Standard deviations', std_group_auc, std_group_pcc])
            group_data.append(['P-value', p_value_auc, p_value_pcc])
            df_group = pd.DataFrame(group_data, columns=['Allele', 'AUC', 'PCC'])
            group_output_path = os.path.join(output_dir, f'best_model_allele_results_fold_{fold + 1}.csv')
            df_group.to_csv(group_output_path, index=False)
            print(f"Allele results for best model in fold {fold + 1} saved to {group_output_path}")
    
            fold_results.append((best_auc, best_pcc))
    
        summary_data = {'Fold': list(range(1, 6)),
                        'AUC': [result[0] for result in fold_results],
                        'PCC': [result[1] for result in fold_results]}
        
        df_summary = pd.DataFrame(summary_data)
        summary_output_path = os.path.join(output_dir, 'cv_summary_best_results.csv')
        df_summary.to_csv(summary_output_path, index=False)
        print(f"Summary of best AUC and PCC for each fold saved to {summary_output_path}")
    
        avg_auc = np.mean([result[0] for result in fold_results])
        avg_pcc = np.mean([result[1] for result in fold_results])
        std_auc = np.std([result[0] for result in fold_results])
        std_pcc = np.std([result[1] for result in fold_results])
    
        print(f"Average AUC across 5 folds: {avg_auc:.4f}, Standard Deviation: {std_auc:.4f}")
        print(f"Average PCC across 5 folds: {avg_pcc:.4f}, Standard Deviation: {std_pcc:.4f}")


if __name__ == "__main__":

    h5_file_path = os.path.expanduser("/home/data/ProtT5.h5")

    dataset = BindDataset(h5_file_path)

    trainer = Trainer(
        dataset=dataset,
        epochs=25,
        lr=0.001,
        seed=315,
    )

    trainer.fivecv()