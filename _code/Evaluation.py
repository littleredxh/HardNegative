import torch

from .Feature import feature
from .Utils import recall, recall2, recall2_batch

def test(Data, dsets_dict, model, epoch, writer, multi_gpu=False, save_dst=None):
        
    # calculate the retrieval accuracy
    if Data=='ICR':
        # test set r@1
        val_fvec = feature(dsets_dict['val'], model, multi_gpu)
        gal_fvec = feature(dsets_dict['gal'], model, multi_gpu)
        acc = recall2(val_fvec,
                      gal_fvec, 
                      dsets_dict['val'].idx_to_class, 
                      dsets_dict['gal'].idx_to_class,
                      rank=[1,10,20,30])
        
        print('R@1 :{:.2f}'.format(acc[0]*100))
        print('R@10:{:.2f}'.format(acc[1]*100))
        print('R@20:{:.2f}'.format(acc[2]*100))
        print('R@30:{:.2f}'.format(acc[3]*100))
        
        writer.add_scalar(Data+'_test_R@1', acc[0], epoch)
        writer.add_scalar(Data+'_test_R@10', acc[1], epoch)
        writer.add_scalar(Data+'_test_R@20', acc[2], epoch)
        writer.add_scalar(Data+'_test_R@30', acc[3], epoch)
        
    elif Data=='HOTEL':
        from .Utils_hotel import label_transform
        # test set r@1   
        acc, pred_top100 = recall2_batch(feature(dsets_dict['val'], model, multi_gpu),
                                         feature(dsets_dict['tra'], model, multi_gpu),
                                         label_transform(dsets_dict['val']),
                                         label_transform(dsets_dict['tra']),
                                         topk=100)

        torch.save(pred_top100, save_dst+str(epoch)+'pred_top100.pth')
        
        print('R@1:{:.4f}'.format(acc)) 

        writer.add_scalar(Data+'_test_R@1', acc, epoch)
        
    elif Data=='SOP':
        # train set r@1
        tra_fvec = feature(dsets_dict['tra'], model, multi_gpu)
        acc_tra = recall(tra_fvec, dsets_dict['tra'].idx_to_class)
        
        # test set r@1
        val_fvec = feature(dsets_dict['val'], model, multi_gpu)
        acc_val = recall(val_fvec, dsets_dict['val'].idx_to_class, rank=[1,10,100,1000])
        
        print('R@1:{:.2f}'.format(acc_val[0]*100)) 
        print('R@10:{:.2f}'.format(acc_val[1]*100))
        print('R@100:{:.2f}'.format(acc_val[2]*100))
        print('R@1000:{:.2f}'.format(acc_val[3]*100))
        
        writer.add_scalar(Data+'_train_R@1', acc_tra, epoch)
        writer.add_scalar(Data+'_test_R@1', acc_val[0], epoch)
        writer.add_scalar(Data+'_test_R@10', acc_val[1], epoch)
        writer.add_scalar(Data+'_test_R@100', acc_val[2], epoch)
        writer.add_scalar(Data+'_test_R@1000', acc_val[3], epoch)
        
    else:
        # train set r@1
        tra_fvec = feature(dsets_dict['tra'], model, multi_gpu)
        acc_tra = recall(tra_fvec, dsets_dict['tra'].idx_to_class)
        writer.add_scalar(Data+'_train_R@1', acc_tra, epoch)
        
        # test set r@1
        val_fvec = feature(dsets_dict['val'], model, multi_gpu)
        acc_val = recall(val_fvec, dsets_dict['val'].idx_to_class, rank=[1,2,4,8])
        writer.add_scalar(Data+'_test_R@1', acc_val[0], epoch)
        
        print('R@1_tra:{:.1f} R@1_val:{:.1f}'.format(acc_tra*100, acc_val[0]*100)) 
        
        writer.add_scalar(Data+'_test_R@1', acc_val[0], epoch)
        writer.add_scalar(Data+'_test_R@2', acc_val[1], epoch)
        writer.add_scalar(Data+'_test_R@4', acc_val[2], epoch)
        writer.add_scalar(Data+'_test_R@8', acc_val[3], epoch)
        
    return