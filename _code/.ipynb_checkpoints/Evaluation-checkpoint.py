import torch

from .Feature import feature
from _code.Utils import recall, recall2

def test(Data, dsets_dict, model, epoch, writer, multi_gpu=False):
    
    if multi_gpu:
        model.module.eval()  # Set model to testing mode
    else:
        model.eval()  # Set model to testing mode
        
    # calculate the retrieval accuracy
    if Data=='ICR':
        # test set r@1
        acc = recall2(feature(dsets_dict['val'], model),
                      feature(dsets_dict['gal'], model), 
                      dsets_dict['val'].idx_to_class, 
                      dsets_dict['gal'].idx_to_class)
        
        print('R@1:{:.2f}'.format(acc)) 
        
        writer.add_scalar(Data+'_test_R@1', acc, epoch)
        
    elif Data=='HOTEL':
        # test set r@1
        acc = recall2(feature(dsets_dict['val'], model),
                      feature(dsets_dict['tra'], model), 
                      dsets_dict['val'].idx_to_class, 
                      dsets_dict['tra'].idx_to_class)
        
        print('R@1:{:.2f}'.format(acc))
        
        writer.add_scalar(Data+'_test_R@1', acc, epoch)
        
    else:
        # train set r@1
        acc_tra = recall(feature(dsets_dict['tra'], model), dsets_dict['tra'].idx_to_class)
        # test set r@1
        acc_val = recall(feature(dsets_dict['val'], model), dsets_dict['val'].idx_to_class)
        
        print('R@1_tra:{:.1f} R@1_val:{:.1f}'.format(acc_tra*100, acc_val*100)) 
        
        writer.add_scalar(Data+'_train_R@1', acc_tra, epoch)
        writer.add_scalar(Data+'_test_R@1', acc_val, epoch)
        
    return