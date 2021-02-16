import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

def fun_CosSim(Mat_A, Mat_B, norm=1, ):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)
    
    D = Mat_A.mm(torch.t(Mat_B))
    D.fill_diagonal_(-norm)
    return D

def Mat(Lvec):
    N = Lvec.size(0)
    Mask = Lvec.repeat(N,1)
    Same = (Mask==Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same#same diff
    
class SCTLoss(Module):
    def __init__(self, method, lam=1):
        super(SCTLoss, self).__init__()
        
        if method=='sct':
            self.sct = True
            self.semi = False
        elif method=='hn':
            self.sct = False
            self.semi = False
        elif method=='shn':
            self.sct = False
            self.semi = True
        else:
            print('loss type is not supported')
            
        self.lam = lam

    def forward(self, fvec, Lvec):
        # number of images
        N = Lvec.size(0)
        
        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)
        
        # Same/Diff label Matting in Similarity Matrix
        Same, Diff = Mat(Lvec.view(-1))
        
        # Similarity Matrix
        CosSim = fun_CosSim(fvec_norm, fvec_norm)
        
        ############################################
        # finding max similarity on same label pairs
        D_detach_P = CosSim.clone().detach()
        D_detach_P[Diff] = -1
        D_detach_P[D_detach_P>0.9999] = -1
        V_pos, I_pos = D_detach_P.max(1)
 
        # valid positive pairs(prevent pairs with duplicated images)
        Mask_pos_valid = (V_pos>-1)&(V_pos<1)

        # extracting pos score
        Pos = CosSim[torch.arange(0,N), I_pos]
        Pos_log = Pos.clone().detach().cpu()
        
        ############################################
        # finding max similarity on diff label pairs
        D_detach_N = CosSim.clone().detach()
        D_detach_N[Same] = -1
        
        # Masking out non-Semi-Hard Negative
        if self.semi:    
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff] = -1
            
        V_neg, I_neg = D_detach_N.max(1)
            
        # valid negative pairs
        Mask_neg_valid = (V_neg>-1)&(V_neg<1)

        # extracting neg score
        Neg = CosSim[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()
        
        # Mask all valid triplets
        Mask_valid = Mask_pos_valid&Mask_neg_valid
        
        # Mask hard/easy triplets
        HardTripletMask = ((Neg>Pos) | (Neg>0.8)) & Mask_valid
        EasyTripletMask = ((Neg<Pos) & (Neg<0.8)) & Mask_valid
        
        # number of hard triplet
        hn_ratio = (Neg>Pos)[Mask_valid].clone().float().mean().cpu()
        
        # triplets
        Triplet_val = torch.stack([Pos,Neg],1)
        Triplet_idx = torch.stack([I_pos,I_neg],1)
        
        Triplet_val_log = Triplet_val.clone().detach().cpu()
        Triplet_idx_log = Triplet_idx.clone().detach().cpu()
        
        # loss
        if self.sct: # SCT setting
            
            loss_hardtriplet = Neg[HardTripletMask].sum()
            loss_easytriplet = -F.log_softmax(Triplet_val[EasyTripletMask,:]/0.1, dim=1)[:,0].sum()
            
            N_hard = HardTripletMask.float().sum()
            N_easy = EasyTripletMask.float().sum()
            
            if torch.isnan(loss_hardtriplet) or N_hard==0:
                loss_hardtriplet, N_hard = 0, 0
                print('No hard triplets in the batch')
                
            if torch.isnan(loss_easytriplet) or N_easy==0:
                loss_easytriplet, N_easy = 0, 0
                print('No easy triplets in the batch')
                
            N = N_easy + N_hard
            if N==0: N=1
            loss = (loss_easytriplet + self.lam*loss_hardtriplet)/N
                
        else: # Standard Triplet Loss setting
            
            loss = -F.log_softmax(Triplet_val[Mask_valid,:]/0.1, dim=1)[:,0].mean()
            
        print('loss:{:.3f} hn_rt:{:.3f}'.format(loss.item(), hn_ratio.item()), end='\r')

        return loss, Triplet_val_log, Triplet_idx_log, hn_ratio