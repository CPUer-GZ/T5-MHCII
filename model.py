import os
import re
import torch
import torch.nn as nn
import torch.nn.init as init

class CNN(nn.Module):
    def __init__(self, input_dim, sequence_length):
        super(CNN, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,out_channels=128,kernel_size=3,padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.35)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,out_channels=128,kernel_size=5,padding='same'),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Dropout(0.35)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,out_channels=256,kernel_size=7,padding='same'),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.35)
        )
        self.conv_4 = nn.Sequential(
            nn.Conv1d(in_channels=input_dim,out_channels=512,kernel_size=9,padding='same'),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.35)
        )
        
        self.self_attention = nn.MultiheadAttention(embed_dim=1024, num_heads=32)
        self.sigmoid = nn.Sigmoid()

        self._initialize_weights()
        
    def forward(self, x,key_padding_mask):
        x = x.transpose(1, 2)  
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)
        x = torch.cat((x1, x2, x3, x4), dim=1) 

        x = x.permute(2, 0, 1) 
        
        key_padding_mask = key_padding_mask.bool()  
        key_padding_mask = ~key_padding_mask  
        x, _ = self.self_attention(x, x, x, key_padding_mask)
        x = x.permute(1, 2, 0)  
        
        x = torch.max(x, dim=2)[0] 
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                torch.nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

class T5ForPepMHCIIRegression(nn.Module):
    def __init__(self):
        super().__init__()

        self.pep_cnn = CNN(input_dim=1024, sequence_length=26)
        self.mhcii_cnn = CNN(input_dim=1024, sequence_length=35)
        self.pep_mhcii_cnn = CNN(input_dim=1024, sequence_length=60)

        self.binding_predict = nn.Sequential(
            nn.Linear(in_features=1024 * 3, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.35),
            
            nn.Linear(in_features=1024, out_features=256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(0.35),
            
            nn.Linear(in_features=256, out_features=64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(0.35),
            
            nn.Linear(in_features=64, out_features=1)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self,
                peptide_hidden_states,peptide_mask,
                mhcii_hidden_states,mhcii_mask,
                combined_hidden_states,combined_mask
               ):

        peptide_encoders = peptide_hidden_states.masked_fill(~peptide_mask.unsqueeze(-1).bool(), 0.0)
        mhcii_encoders = mhcii_hidden_states.masked_fill(~mhcii_mask.unsqueeze(-1).bool(), 0.0)
        combined_encoders = combined_hidden_states.masked_fill(~combined_mask.unsqueeze(-1).bool(), 0.0)

        pep_cnn_out = self.pep_cnn(peptide_encoders,peptide_mask) 
        mhcii_cnn_out = self.mhcii_cnn(mhcii_encoders,mhcii_mask)  
        pep_mhcii_cnn_out = self.pep_mhcii_cnn(combined_encoders,combined_mask)  

        cnn_out = torch.cat((pep_cnn_out, mhcii_cnn_out, pep_mhcii_cnn_out), dim=-1)  

        output = self.binding_predict(cnn_out)  
        output = self.sigmoid(output)
        return output