import torch
from torch.utils.data.sampler import SequentialSampler
import torch.nn.functional as F
from .Model import setModelMode

def feature(dsets, model, multi_gpu=False):
    Fvecs = []
    dataLoader = torch.utils.data.DataLoader(dsets, batch_size=400, sampler=SequentialSampler(dsets), num_workers=32)
    with torch.set_grad_enabled(False):
        setModelMode(model, 'eva', multi_gpu)
        for data in dataLoader:
            inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
            fvec = model(inputs_bt.cuda())
            fvec = F.normalize(fvec, p = 2, dim = 1).cpu()
            Fvecs.append(fvec)
            
    return torch.cat(Fvecs,0)

from .Model import setModel
from .Reader import ImageReader
from .Utils import eva_transforms
from .color_lib import RGBmean, RGBstdv

def featuralize(Data, data_dict, src_model, model_name, emb_dim):
    dsets = ImageReader(data_dict, eva_transforms(256, RGBmean[Data], RGBstdv[Data]))
    model = setModel(model_name, emb_dim)
    model_state_dict = torch.load(src_model+'model_state_dict.pth')
    model.load_state_dict(model_state_dict)
    fvec = feature(dsets, model.cuda())
    return fvec, dsets
