import torch
from torch.nn.modules.module import Module
import torch.nn as nn
import torch.nn.functional as F

def distDP(Mat_A, Mat_B, norm=1, sq=True):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)
    
    DC = Mat_A.mm(torch.t(Mat_B))
    if sq:
        DC.fill_diagonal_(-norm)
            
    return DC

def Mat(Lvec):
    N = Lvec.size(0)
    Mask = Lvec.repeat(N,1)
    Same = (Mask==Mask.t())
    return Same.clone().fill_diagonal_(0), ~Same#same diff
    
class EPHNLoss(Module):
    def __init__(self, sct=True, semi=False, lam=1):
        super(EPHNLoss, self).__init__()
        self.sct = sct
        self.semi = semi
        self.lam = lam

    def forward(self, fvec, Lvec):
        # number of images
        N = Lvec.size(0)
        
        # feature normalization
        fvec_norm = F.normalize(fvec, p = 2, dim = 1)
        
        # matting
        Same, Diff = Mat(Lvec.view(-1))
        
        # Similarity Matrix
        DotProd = distDP(fvec_norm,fvec_norm)
        
        ############################################
        # finding max similarity on same label pairs
        D_detach_P = DotProd.clone().detach()
        D_detach_P[Diff] = -1
        D_detach_P[D_detach_P>0.9999] = -1
        V_pos, I_pos = D_detach_P.max(1)
 
        # valid positive pairs(prevent pairs with duplicated images)
        Mask_pos_valid = (V_pos>-1)&(V_pos<1)

        # extracting pos score
        Pos = DotProd[torch.arange(0,N), I_pos]
        Pos_log = Pos.clone().detach().cpu()
        
        ############################################
        # finding max similarity on diff label pairs
        D_detach_N = DotProd.clone().detach()
        D_detach_N[Same] = -1
        if self.semi:
            # extracting Semi-Hard Negative
            D_detach_N[(D_detach_N>(V_pos.repeat(N,1).t()))&Diff] = -1
        V_neg, I_neg = D_detach_N.max(1)
            
        # valid negative pairs
        Mask_neg_valid = (V_neg>-1)&(V_neg<1)

        # extracting neg score
        Neg = DotProd[torch.arange(0,N), I_neg]
        Neg_log = Neg.clone().detach().cpu()
        
        # Mask all valid triplets
        Mask_valid = Mask_pos_valid&Mask_neg_valid
        
        # This mask can make the optimization more consistent
        # Mask hard/easy triplets
        HardTripletMask = (Neg>Pos) & (Neg>0.5) & Mask_valid
        EasyTripletMask = (~HardTripletMask) & Mask_valid
        
#         # Mask hard/easy triplets
#         HardTripletMask = (Neg>Pos) & Mask_valid
#         EasyTripletMask = (Neg<Pos) & Mask_valid
        
        # number of hard triplet
        hn_ratio = (Neg>Pos)[Mask_valid].clone().float().mean().cpu()
        
        # triplets
        Triplet_val = torch.stack([Pos,Neg],1)
        Triplet_idx = torch.stack([I_pos,I_neg],1)
        
        # loss
        if self.sct: # SCT setting
            loss_hardtriplet = Neg[HardTripletMask].mean()
            loss_easytriplet = -F.log_softmax(Triplet_val[EasyTripletMask,:]/0.1,dim=1)[:,0].mean()
            
            if torch.isnan(loss_easytriplet) and not torch.isnan(loss_hardtriplet):
                loss = loss_hardtriplet*self.lam
                print('HT only')
                
            if torch.isnan(loss_hardtriplet) and not torch.isnan(loss_easytriplet):
                loss = loss_easytriplet
                print('ET only')
            
            if not torch.isnan(loss_hardtriplet) and not torch.isnan(loss_easytriplet):
                loss = loss_easytriplet + loss_hardtriplet*self.lam
                
            if torch.isnan(loss_hardtriplet) and torch.isnan(loss_easytriplet):
                print('nan loss')
                loss = loss_hardtriplet
                
        else:
            loss = -F.log_softmax(Triplet_val[Mask_valid,:]/0.1,dim=1)[:,0].mean()
            
        print('loss:{:.3f} hn_rt:{:.3f}'.format(loss.item(), hn_ratio.item()), end='\r')

        return loss, Triplet_val.clone().detach().cpu(), Triplet_idx.clone().detach().cpu(), hn_ratio