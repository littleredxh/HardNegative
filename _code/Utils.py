import torch
from torchvision import transforms

def tra_transforms(imgsize, RGBmean, RGBstdv):
    return transforms.Compose([transforms.Resize(int(imgsize*1.1)),
                                 transforms.RandomCrop(imgsize),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(RGBmean, RGBstdv)])

def eva_transforms(imgsize, RGBmean, RGBstdv):
    return transforms.Compose([transforms.Resize(imgsize),
                                 transforms.CenterCrop(imgsize),
                                 transforms.ToTensor(),
                                 transforms.Normalize(RGBmean, RGBstdv)])

def RP(D,imgLab,label_counts):
    # for each image calculate R-Precision
    A=0
    for i in range(D.size(1)):
        # Find R nearest neighbors index
        _,idx = D[i,:].topk(label_counts[i])
        # Convert R nearest neighbors index to label
        imgPre = imgLab[idx]
        # compare R nearest neighbors label to ground truth label
        # and calculate the R-Precision
        A += (imgPre==imgLab[i]).float().mean()

    return (A/D.size(1)).item()

def MAPR(D,imgLab,label_counts):
    # for each image calculate R-Precision
    A=0
    for i in range(D.size(1)):
        # Find R nearest neighbors index
        _,idx = D[i,:].topk(label_counts[i])
        w = torch.arange(1,label_counts[i]+1)
        # Convert R nearest neighbors index to label
        imgPre = imgLab[idx]
        # compare R nearest neighbors label to ground truth label
        # and calculate the R-Precision
        p = (imgPre==imgLab[i]).float()
        p = p*p.cumsum(0)
        A += (p/w).mean()

    return (A/D.size(1)).item()

def recall(Fvec, imgLab, rank=None):
    N = len(imgLab)
    imgLab = torch.LongTensor([imgLab[i] for i in range(len(imgLab))])
    
    D = Fvec.mm(torch.t(Fvec))
    D[torch.eye(len(imgLab)).bool()] = -1
    
    if rank==None:
        _,idx = D.max(1)
        imgPre = imgLab[idx]
        A = (imgPre==imgLab).float()
        return (torch.sum(A)/N).item()
    else:
        _,idx = D.topk(rank[-1])
        acc_list = []
        for r in rank:
            A = 0
            for i in range(r):
                imgPre = imgLab[idx[:,i]]
                A += (imgPre==imgLab).float()
            acc_list.append((torch.sum((A>0).float())/N).item())
        return acc_list
    
def recall2(Fvec_val, Fvec_gal, imgLab_val, imgLab_gal, rank=None):
    N = len(imgLab_val)
    imgLab_val = torch.LongTensor([imgLab_val[i] for i in range(len(imgLab_val))])
    imgLab_gal = torch.LongTensor([imgLab_gal[i] for i in range(len(imgLab_gal))])
    
    D = Fvec_val.mm(torch.t(Fvec_gal))
    
    if rank==None:
        _,idx = D.max(1)
        imgPre = imgLab_gal[idx]
        A = (imgPre==imgLab_val).float()
        return (torch.sum(A)/N).item()
    else:
        _,idx = D.topk(rank[-1])
        acc_list = []
        for r in rank:
            A = 0
            for i in range(r):
                imgPre = imgLab_gal[idx[:,i]]
                A += (imgPre==imgLab_val).float()
            acc_list.append((torch.sum((A>0).float())/N).item())
        return torch.Tensor(acc_list)
    
def genInterval(Len, gap):
    N = Len//gap
    interval=[]
    for i in range(N):
        interval.append([i*gap,(i+1)*gap])
    if Len%gap!=0:
        interval.append([(i+1)*gap,Len])
    
    return interval

def recall2_batch(Fvec_val, Fvec_gal, imgLab_val, imgLab_gal, topk=100, gap=100):
    N = len(imgLab_val)
    imgLab_val = torch.LongTensor([imgLab_val[i] for i in range(len(imgLab_val))])
    imgLab_gal = torch.LongTensor([imgLab_gal[i] for i in range(len(imgLab_gal))])
    
    interval_out = genInterval(N, gap)
    idx=[]
    for itv in interval_out:
        D = (Fvec_val[itv[0]:itv[1],:].cuda()).mm(torch.t(Fvec_gal).cuda())
        idx.append(D.sort(1,descending=True)[1][:,:topk].cpu())
        print('{:.2f}'.format(itv[0]/N*100),end='\r')
    idx = torch.cat(idx,0)
    print(idx.size())
    imgPre = imgLab_gal[idx[:,0]]
    A = (imgPre==imgLab_val).float()
    return (torch.sum(A)/N).item(), idx
    
def RunAcc(src, rank, phase='val'):
    Fvec = torch.load(src +'39'+ phase + 'Fvecs.pth')
    dsets = torch.load(src + phase + 'dsets.pth')
    
    acc = recall(Fvec, dsets.idx_to_class,rank=rank)
    
    torch.save(acc,src+'acc.pth')
    torch.set_printoptions(precision=1)
    print(acc*100)
    torch.set_printoptions(precision=3)
    return acc

def RunAcc2(src, rank):
    Fvec_val = torch.load(src+'valFvecs.pth')
    Fvec_gal = torch.load(src+'galFvecs.pth')
    dsets_val = torch.load(src+'valdsets.pth')
    dsets_gal = torch.load(src+'galdsets.pth')
    
    acc = recall2(Fvec_val, Fvec_gal, dsets_val.idx_to_class, dsets_gal.idx_to_class,rank=rank)

    torch.save(acc,src+'acc.pth')
    torch.set_printoptions(precision=1)
    print(acc*100)
    torch.set_printoptions(precision=3)
    return acc

def norml2(vec):# input N by F
    F = vec.size(1)
    w = torch.sqrt((torch.t(vec.pow(2).sum(1).repeat(F,1))))
    return vec.div(w)

def distM2(Mat_A,Mat_B):#N by F
    N_A = Mat_A.size(0)
    N_B = Mat_B.size(0)
    
    A1 = Mat_A
    B1 = Mat_B
    
    A2 = Mat_A.pow(2).sum(1).repeat(N_B,1)# N_B by N_A
    B2 = Mat_B.pow(2).sum(1).repeat(N_A,1)# N_A by N_B

    D2 = torch.t(A2) - 2*A1.mm(torch.t(B1)) + B2
        
    D2 = torch.sqrt(D2)
    # D2[D2!=D2]=eps
    return D2