import os, time, torch
import argparse
import sys
sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"]="1"

from torch.utils.tensorboard import SummaryWriter

from _code.color_lib import RGBmean, RGBstdv
from _code.Model import setModel, setOptimizer, setModelMode
from _code.Sampler import BalanceSampler_sample, BalanceSampler_filled
from _code.Reader import ImageReader
from _code.Utils import tra_transforms, eva_transforms
from _code.Evaluation import test
from _code.Loss import SCTLoss

parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: CUB, CAR, SOP or ICR')
parser.add_argument('--i', type=int, help='index of experiment')
parser.add_argument('--model', type=str, help='backbone model: R18 or R50')
parser.add_argument('--dim', type=int, help='embedding dimension')
parser.add_argument('--lr', type=float, help='initial learning rate')
parser.add_argument('--epochs', type=int, help='epochs')
parser.add_argument('--ngpu', type=int, help='number of gpu')
parser.add_argument('--bsize', type=int, help='batch size')
parser.add_argument('--repeat', type=int, help='dataset repeats times')
parser.add_argument('--lam', type=float, help='lam')
parser.add_argument('--method', type=str, help='loss function: sct, hn or shn')
args = parser.parse_args()

Data = args.Data
Id = args.i
Model = args.model
Dim = args.dim
Lr = args.lr
Epochs = args.epochs
Ngpu = args.ngpu
Bsize = args.bsize
Repeat = args.repeat
Lam = args.lam
Method = args.method

    
##############################
# save dir
dst = '_result/{}_{}/{}_B{}_R{}_D{}_lr{}_lam{}/{}/'.format(Data, Model, Method, Bsize, Repeat, Dim, Lr, Lam, Id)
print('saving directory: {}'.format(dst))
if os.path.exists(dst): 
    print('experiment existed')
    sys.exit(0)

# data dict
data_dict = torch.load('/home/xuanhong/datasets/{}/data_dict_emb.pth'.format(Data))
phase = data_dict.keys()

# dataset setting
imgsize = 256
tra_transform = tra_transforms(imgsize, RGBmean[Data], RGBstdv[Data])
eva_transform = eva_transforms(imgsize, RGBmean[Data], RGBstdv[Data])

# sampler setting
N_size = 2
C_size = Bsize//N_size
num_workers = 32
print('batch size: {}'.format(Bsize))
print('number of images per class: {}'.format(N_size))
print('number of classes per batch: {}'.format(C_size))

# loss setting
criterion = SCTLoss(Method, lam=args.lam) 

# recorder setting
writer = SummaryWriter(dst)
test_freq = 10

# model setting
model = setModel(Model, Dim).cuda()
print('output dimension: {}'.format(Dim))

# Multi_GPU setting
multi_gpu = (Ngpu>1)
if multi_gpu:
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(Ngpu)], output_device=0)

# Optimizer and scheduler setting
optimizer, scheduler = setOptimizer(model.parameters(), args.lr, [int(Epochs*0.5),])


##############################
# training
since = time.time() # recording time
global_it = 0
for epoch in range(Epochs+1): 

    print('Epoch {}/{} \n '.format(epoch, Epochs) + '-' * 40)

    # train phase
    if epoch>0:  
        # create dset
        dsets = ImageReader(data_dict['tra'], tra_transform) 

        # create sampler
        if Data in ['SOP','ICR','HOTEL']:
            sampler = BalanceSampler_sample(dsets.intervals, GSize=N_size, repeat=args.repeat)
        else:
            sampler = BalanceSampler_filled(dsets.intervals, GSize=N_size, repeat=args.repeat)

        # create dataloader
        dataLoader = torch.utils.data.DataLoader(dsets, batch_size=Bsize, sampler=sampler, num_workers=num_workers)
        
        # Set model to training mode
        setModelMode(model, 'tra', multi_gpu)
 
        # record loss
        L_data, N_data = 0.0, 0

        # iterate batch
        for data in dataLoader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                if len(labels_bt)<Bsize: continue
                fvec = model(inputs_bt.cuda())
                loss, _, _, hn_ratio = criterion(fvec, labels_bt.cuda())
                
                if torch.isnan(loss): 
                    print('loss is nan')
                    continue
                    
                loss.backward()
                optimizer.step() 
                
            writer.add_scalar('hn_ratio', hn_ratio, global_it)
            global_it+=1

            L_data += loss.item()
            N_data += 1
            
        writer.add_scalar('loss', L_data/N_data, epoch)
        # adjust the learning rate
        scheduler.step()
    
    # evaluation phase
    if epoch%test_freq==0:
        # evaluate train set
        dsets_dict = {p: ImageReader(data_dict[p], eva_transform) for p in phase}
        acc = test(Data, dsets_dict, model, epoch, writer, multi_gpu=multi_gpu)

##############################
# save model
if multi_gpu:
    torch.save(model.module.cpu().state_dict(), dst + 'model_state_dict.pth')
else:
    torch.save(model.cpu().state_dict(), dst + 'model_state_dict.pth')
    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))


