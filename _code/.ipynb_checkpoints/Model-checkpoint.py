from torchvision import models
import torch.optim as optim
import torch.nn as nn

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

def setOptimizer(parameters, init_lr, milestones):
    print('LR is set to {}'.format(init_lr))
    optimizer = optim.SGD(parameters, lr=init_lr, momentum=0.0)
#     optimizer = optim.RMSprop(parameters, lr=init_lr, momentum=0.0)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1)

    return optimizer, scheduler

#     optimizer = torch.optim.Adam(self.parameters(), lr=self.max_lr)
#     scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, self.max_lr,
#     cycle_momentum=False,
#     epochs=self.n_epochs,
#     steps_per_epoch=int(np.ceil(len(X) / self.batch_size)),
#     )