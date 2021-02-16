from torch.utils.data.sampler import Sampler
import numpy as np
import random
import torch

class BalanceSampler_filled(Sampler):
    def __init__(self, intervals, GSize=2, repeat=1):
        
        class_len = len(intervals)
        list_sp = []
        self.idx = []
        
        # find the max interval
        interval_list = [np.arange(b[0],b[1]) for b in intervals]
        len_max = max([b[1]-b[0] for b in intervals])

        # exact division
        if len_max%GSize != 0:
            if len_max%GSize<int(0.3*GSize):
                len_max = len_max-len_max%GSize
            else:
                len_max = len_max-len_max%GSize+GSize
            
        for _ in range(repeat):
            # filled images for each class
            for l in interval_list:
                if l.shape[0]<len_max:
                    l_ext = np.random.choice(l,len_max-l.shape[0])
                    l_ext = np.concatenate((l, l_ext), axis=0)
                    l_ext = np.random.permutation(l_ext)
                elif l.shape[0]>len_max:
                    l_ext = np.random.choice(l,len_max,replace=False)
                    l_ext = np.random.permutation(l_ext)
                elif l.shape[0]==len_max:
                    l_ext = np.random.permutation(l)

                list_sp.append(l_ext)

            random.shuffle(list_sp)
            self.idx += np.vstack(list_sp).reshape((GSize*class_len,-1)).T.reshape((1,-1)).flatten().tolist()
        print('total images size in sampler: {}'.format(len(self.idx)))
        
        
    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)
    

class BalanceSampler_sample(Sampler):
    def __init__(self, intervals, GSize=2, repeat=1):
        # generate interval list
        self.idx = []
        for _ in range(repeat):
            interval_list = []
            for b in intervals:
                index_list = torch.arange(b[0],b[1]).tolist()
                if b[1]-b[0]>GSize:
                    interval_list.append(random.sample(index_list,GSize))
                else:
                    interval_list.append(index_list)

            random.shuffle(interval_list)

            for l in interval_list:
                self.idx += l

    def __iter__(self):
        return iter(self.idx)
    
    def __len__(self):
        return len(self.idx)