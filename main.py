import os, time, torch, sys
import argparse

from torch.utils.tensorboard import SummaryWriter

from _code.color_lib import RGBmean, RGBstdv
from _code.Loss import EPHNLoss
from _code.Model import setModel, setOptimizer
from _code.Sampler import BalanceSampler_sample, BalanceSampler_filled
from _code.Reader import ImageReader
from _code.Utils import tra_transforms, eva_transforms
from _code.Evaluation import test

parser = argparse.ArgumentParser(description='running parameters')
parser.add_argument('--Data', type=str, help='dataset name: CUB, CAR, SOP or ICR')
parser.add_argument('--model', type=str, help='backbone model: R18 or R50')
parser.add_argument('--dim', type=int, help='embedding dimension')
parser.add_argument('--lr', type=float, help='initial learning rate')
parser.add_argument('--epochs', type=int, help='epochs')
parser.add_argument('--ngpu', type=int, help='number of gpu')
parser.add_argument('--i', type=int, help='index of experiment')
parser.add_argument('--semi', type=int, help='semi-hard?')
parser.add_argument('--sct', type=int, help='sct?')
parser.add_argument('--bsize', type=int, help='batch size')
parser.add_argument('--lam', type=float, help='lam')
parser.add_argument('--repeat', type=int, help='dataset repeats')
args = parser.parse_args()

if args.semi==0 and args.sct==0:
    method='HN'
elif args.semi==1 and args.sct==0:
    method='SHN'
elif args.semi==0 and args.sct==1:
    method='SCT'
else:
    print('method is not supported')
    sys.exit(0)
    
##############################
# data dict
Data = args.Data
dst = '{}_{}_result_r{}_lam{}/{}/D{}_lr{}/{}/'.format(args.Data, args.model, args.repeat, args.lam, method, args.dim, args.lr, args.i)

data_dict = torch.load('/home/xuanhong/datasets/{}/data_dict_emb.pth'.format(Data))
phase = data_dict.keys()
print('saving directory: {}'.format(dst))

# dataset setting
imgsize = 256
tra_transform = tra_transforms(imgsize, RGBmean[Data], RGBstdv[Data])
eva_transform = eva_transforms(imgsize, RGBmean[Data], RGBstdv[Data])

# network setting
model_name = args.model
emb_dim = args.dim
multi_gpu = True
n_gpu = args.ngpu

# sampler setting
batch_size = args.bsize
num_workers = 16
print('batch size: {}'.format(batch_size))

# loss setting
criterion = EPHNLoss(sct=bool(args.sct), semi=bool(args.semi), lam=args.lam) 
N_size = 2
C_size = int(batch_size/N_size)
print('number of images per class: {}'.format(N_size))
print('number of classes per batch: {}'.format(C_size))

# recorder frequency
num_epochs = args.epochs
test_freq = 5
writer = SummaryWriter(dst)

# model setting
model = setModel(model_name, emb_dim).cuda()
print('output dimension: {}'.format(emb_dim))

if multi_gpu:
    model = torch.nn.DataParallel(model, device_ids=[i for i in range(n_gpu)], output_device=0)

# Optimizer and scheduler setting
optimizer, scheduler = setOptimizer(model.parameters(), args.lr, [int(num_epochs*0.5), int(num_epochs*0.75)])


##############################
# training
since = time.time() # recording time
global_it = 0
for epoch in range(num_epochs+1): 

    print('Epoch {}/{} \n '.format(epoch, num_epochs) + '-' * 40)

    # train phase
    if epoch>=0:  
        # create dset
        dsets = ImageReader(data_dict['tra'], tra_transform) 

        # create sampler
        if Data in ['SOP','ICR']:
            sampler = BalanceSampler_sample(dsets.intervals, GSize=N_size, repeat=args.repeat)
        else:
            sampler = BalanceSampler_filled(dsets.intervals, GSize=N_size, repeat=args.repeat)

        # create dataloader
        dataLoader = torch.utils.data.DataLoader(dsets, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        
        # Set model to training mode
        if multi_gpu:
            model.module.train()
        else:
            model.train()
 
        # record loss
        L_data, N_data = 0.0, 0

        # iterate batch
        for data in dataLoader:
            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                inputs_bt, labels_bt = data # <FloatTensor> <LongTensor>
                fvec = model(inputs_bt.cuda())
                loss, Triplet_val, Triplet_idx, hn_ratio = criterion(fvec, labels_bt.cuda())
                if torch.isnan(loss): continue
                loss.backward()
                optimizer.step() 
                
            writer.add_scalar('hn_ration', hn_ratio, global_it)
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
        acc = test(Data, dsets_dict, model, epoch, writer, multi_gpu=True)

##############################
# save model
if multi_gpu:
    torch.save(model.module.cpu().state_dict(), dst + 'model_state_dict.pth')
else:
    torch.save(model.cpu().state_dict(), dst + 'model_state_dict.pth')
    
time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed//60, time_elapsed%60))


