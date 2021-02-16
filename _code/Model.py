from torchvision import models
import torch.optim as optim
import torch.nn as nn

def setModelMode(model, mode, multi_gpu=False):
    if mode=='tra': # Set model to training mode
        if multi_gpu:
            model.module.train()
        else:
            model.train()
    elif mode=='eva': # Set model to evaluation mode
        if multi_gpu:
            model.module.eval()
        else:
            model.eval()
    else:
        raise ValueError('Unrecognized mode')
            
def setModel(model_name, out_dim):
    
    if model_name == 'R18':
        model = models.resnet18(pretrained=True)
        print('Setting model: resnet18')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_dim)
        
    elif model_name == 'R50':
        model = models.resnet50(pretrained=True)
        print('Setting model: resnet50')
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_dim)
        
    elif model_name == 'GBN':
        model = models.googlenet(pretrained=True, transform_input=False)
        model.aux_logits=False
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, out_dim)
        print('Setting model: GoogleNet')
        
    else:
        print('model is not exited!')
    
    return model

def setOptimizer(parameters, init_lr, milestones, step=0.1):
    print('LR is set to {}'.format(init_lr))
    optimizer = optim.SGD(parameters, lr=init_lr, momentum=0.0)
#     optimizer = optim.RMSprop(parameters, lr=init_lr, momentum=0.0)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=step)

    return optimizer, scheduler