import time
import torch
import numpy as np
from scipy.stats import wasserstein_distance as dist
from scipy.stats import ks_2samp
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from . data_loaders import DataGen, SimpleSampler, collate_fn

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def train_one_epoch(epoch_id,
                    net,criterion, optimizer,
                    clf_loader, data_type):
    net.train()
    epoch_loss_clf = 0
    accuracy_clf   = 0
    datalen = len(clf_loader)
    loop = range(0, datalen)
    clf_iter = iter(clf_loader)
    for i in loop:
        net.zero_grad()
        if data_type=='sparse':
            coordins_batch_MC, features_batch_MC, y_batch_clf_MC, events_batch_MC = next(clf_iter)
            x_clf_MC = coordins_batch_MC, features_batch_MC.cuda(), y_batch_clf_MC.shape[0]
        elif data_type=='dense':
            x_clf_MC, y_batch_clf_MC, events_batch_MC = next(clf_iter)
            x_clf_MC = x_clf_MC.cuda()
        y_clf_MC = y_batch_clf_MC.cuda()
        out_clf_MC  = net(x_clf_MC)
        out_clf_MC = out_clf_MC.float()
        clf_loss = criterion(out_clf_MC, y_clf_MC)
        loss = clf_loss
        loss.backward()
        optimizer.step()
        y_pred_clf      = out_clf_MC  .argmax(dim=-1)
        acc_step = (y_pred_clf == y_clf_MC).squeeze().sum().item()/len(y_pred_clf)
        accuracy_clf += acc_step
        epoch_loss_clf += clf_loss.item()
    epoch_loss_clf /= (i+1)
    accuracy_clf /= (i+1)
    print("train  {:5d}: loss_clf:  {:.9f} acc_clf: {:.9f} ".format(epoch_id, epoch_loss_clf, accuracy_clf))
    clf_loader.sampler.on_epoch_end()
    return (epoch_loss_clf, accuracy_clf)

def evaluate_valid(epoch_id, net, criterion, clf_loader, data_type):
    net.eval()
    print('**********EVAL**************')
    epoch_loss_clf = 0
    accuracy_clf   = 0
    datalen = len(clf_loader)
    loop = range(0, datalen)
    clf_iter = iter(clf_loader)
    with torch.autograd.no_grad():
        for i in loop:
            if data_type=='sparse':
                coordins_batch_MC, features_batch_MC, y_batch_clf_MC, events_batch_MC = next(clf_iter)
                x_clf_MC = coordins_batch_MC, features_batch_MC.cuda(), y_batch_clf_MC.shape[0]
            elif data_type=='dense':
                x_clf_MC, y_batch_clf_MC, events_batch_MC = next(clf_iter)
                x_clf_MC = x_clf_MC.cuda()
            y_clf_MC = y_batch_clf_MC.cuda()
            out_clf_MC  = net(x_clf_MC)
            out_clf_MC = out_clf_MC.float()
            
            clf_loss = criterion(out_clf_MC, y_clf_MC)
                        
            y_pred_clf      = out_clf_MC  .argmax(dim=-1)
            accuracy_clf += (y_pred_clf == y_clf_MC).squeeze().sum().item()/len(y_pred_clf)
            epoch_loss_clf += clf_loss.item()
    
        epoch_loss_clf /= (i+1)
        accuracy_clf /= (i+1)
        print("valid  {:5d}: loss_clf:  {:.9f} acc_clf: {:.9f} ".format(epoch_id, epoch_loss_clf, accuracy_clf))
    clf_loader.sampler.on_epoch_end()
    print('*********************')
    return (epoch_loss_clf, accuracy_clf)

def calculate_distance(epoch_id, net, MC_df_below, MC_df_above, data_df_below, data_df_above, data_type, batch_size, num_workers, q):
    MC_prediction_below, MC_evs_below = predict(net,
                                                MC_df_below,
                                                data_type,
                                                batch_size = batch_size,
                                                num_workers = num_workers,
                                                q = q)
    MC_prediction_above, MC_evs_above = predict(net,
                                                MC_df_above,
                                                data_type,
                                                batch_size = batch_size,
                                                num_workers = num_workers,
                                                q = q)
    
    data_prediction_below, data_evs_below = predict(net,
                                                    data_df_below,
                                                    data_type,
                                                    batch_size = batch_size,
                                                    num_workers = num_workers,
                                                    q = q)
    
    data_prediction_above, data_evs_above = predict(net,
                                                    data_df_above,
                                                    data_type,
                                                    batch_size = batch_size,
                                                    num_workers = num_workers,
                                                    q = q)
    dist_below = ks_2samp(MC_prediction_below, data_prediction_below)
    dist_above = ks_2samp(MC_prediction_above, data_prediction_above)
    print("Distribution pval  {:5d}: below:  {:.9f} above: {:.9f} ".format(epoch_id, dist_below.pvalue, dist_above.pvalue))
    return dist_below, dist_above


def train(*,
          net,
          data_type,
          criterion,
          optimizer,
          scheduler,
          batch_size,
          nb_epoch,
          train_df,
          valid_df,
          data_df,
          selection_df,
          model_path,
          tensorboard_dir = '/logs/',
          save_loss   = 0.5,
          num_workers = 0,
          q           = 0,
          augmentation = True,
          ):
    writer = SummaryWriter(model_path+tensorboard_dir)
    datagen_clf        = DataGen(train_df, data_type, q=q, augmentation=augmentation,  node='lowTh')
    datagen_clf_valid  = DataGen(valid_df, data_type, q=q, augmentation=False, node='lowTh')
    
    sampler_clf        = SimpleSampler(datagen_clf)
    sampler_clf_valid  = SimpleSampler(datagen_clf_valid)

    clf_loader = torch.utils.data.DataLoader(datagen_clf, sampler=sampler_clf, batch_size=batch_size, 
                                             shuffle=False, num_workers=num_workers, collate_fn=collate_fn(data_type), drop_last=True, pin_memory=False)
        
    clf_valid_loader = torch.utils.data.DataLoader(datagen_clf_valid, sampler=sampler_clf_valid, batch_size=batch_size, 
                                                   shuffle=False, num_workers=num_workers, collate_fn=collate_fn(data_type), drop_last=True, pin_memory=False)

    
    MC_df_below = selection_df[(selection_df.domain=='MC') & (selection_df.sideband=='lower')]
    MC_df_below = MC_df_below.merge(valid_df, on = ['event', 'run_number'], how='inner').reset_index(drop=True)
    MC_df_above = selection_df[(selection_df.domain=='MC') & (selection_df.sideband=='higher')]
    MC_df_above = MC_df_above.merge(valid_df, on = ['event', 'run_number'], how='inner').reset_index(drop=True)


    data_df_below = selection_df[(selection_df.domain=='data') & (selection_df.sideband=='lower')]
    data_df_below = data_df_below.merge(data_df, on = ['event', 'run_number'], how='inner').reset_index(drop=True)
    data_df_above = selection_df[(selection_df.domain=='data') & (selection_df.sideband=='higher')]
    data_df_above = data_df_above.merge(data_df, on = ['event', 'run_number'], how='inner').reset_index(drop=True)

    for i in range(0, nb_epoch):
        t0 = time.time()

        train_stats = train_one_epoch(i, net,
                                      criterion,
                                      optimizer,
                                      clf_loader,
                                      data_type)
        evaluate_stats = evaluate_valid(i, net, criterion, clf_valid_loader, data_type)
        dist_below, dist_above = calculate_distance(i, net, MC_df_below, MC_df_above, 
                                                    data_df_below, data_df_above, data_type, batch_size, num_workers, q)
        writer.add_scalar('Loss_clf/train', train_stats[0], i)
        writer.add_scalar('Loss_clf/test', evaluate_stats[0], i)
        writer.add_scalar('Accuracy_clf/train', train_stats[1], i)
        writer.add_scalar('Accuracy_clf/test', evaluate_stats[1], i)
        writer.add_scalar('KS/below', dist_below[0], i)
        writer.add_scalar('KS/above', dist_above[0], i)
        writer.add_scalar('KS/p_below', dist_below[1], i)
        writer.add_scalar('KS/p_above', dist_above[1], i)

        if evaluate_stats[0] < save_loss:
            best_loss, best_acc = evaluate_stats
            filename = model_path+'check_point_epoch_{}_loss_{}_acc_{}'.format(i, round(best_loss,2), round(best_acc,2))
            save_checkpoint({
                'epoch': i,
                'state_dict': net.state_dict(),
                'best_loss': best_loss,
                'best_acc' : best_acc,
                'optimizer' : optimizer.state_dict(),
            }, filename)


def predict(
        net,
        test_df,
        data_type,
        batch_size = 1024,
        num_workers = 0,
        q = 0):
    net.eval()

    datagen_test = DataGen(test_df, data_type, q=q, node='lowTh')
    testloader = torch.utils.data.DataLoader(datagen_test,  batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers,
                                             collate_fn=collate_fn(data_type), drop_last=False, pin_memory=False)
    
    prediction = np.zeros(len(test_df))
    evs = np.zeros(len(test_df))
    idx = 0
    for batch in testloader:
        with torch.autograd.no_grad():
            if data_type=='sparse':
                coordins_batch, features_batch, y_batch_clf, events_batch = batch
                x_clf = coordins_batch, features_batch.cuda(), y_batch_clf.shape[0]
            elif data_type=='dense':
                x_batch_clf, y_batch_clf, events_batch = batch
                x_clf = x_batch_clf.float().cuda()
            out = nn.functional.softmax(net(x_clf).float(), dim=1)[:,1]
            y_pred = out.cpu().detach().numpy()
            prediction[idx : idx+len(y_pred)] = y_pred
            evs[idx : idx+len(y_pred)] = events_batch
            idx+=len(y_pred)
    return prediction, evs

