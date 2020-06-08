import time
import torch
import numpy as np
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from . data_loaders import DataGen, SimpleSampler, BalancedSampler, collate_fn

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def train_one_epoch(epoch_id,
                    net,
                    criterion,
                    optimizer,
                    clf_loader,
                    data_loader,
                    data_type,
                    nb_epoch):
    net.train()
    epoch_loss_clf = 0
    accuracy_clf   = 0
    epoch_loss_dom = 0
    accuracy_dom   = 0
    datalen = min(len(clf_loader), len(data_loader))
    loop = range(0, datalen)
    clf_iter = iter(clf_loader)
    data_iter = iter(data_loader)
    for i in loop:
        p = float(i + epoch_id * datalen) /nb_epoch / datalen
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        net.zero_grad()
        if data_type=='sparse':
            coordins_batch_MC, features_batch_MC, y_batch_clf_MC, events_batch_MC = next(clf_iter)
            x_clf_MC = coordins_batch_MC, features_batch_MC.cuda(), y_batch_clf_MC.shape[0]
        elif data_type=='dense':
            x_batch_clf_MC, y_batch_clf_MC, events_batch_MC = next(clf_iter)
            x_clf_MC = x_batch_clf_MC.float().cuda()
        y_clf_MC = y_batch_clf_MC.cuda()
        out_clf_MC, out_dom_MC  = net(x_clf_MC, alpha)
        out_clf_MC = out_clf_MC.float()
        clf_loss = criterion(out_clf_MC, y_clf_MC)
        if data_type=='sparse':
            coordins_batch_data, features_batch_data, y_batch_data, events_batch_data = next(data_iter)
            x_dom_data = coordins_batch_data, features_batch_data.cuda(), y_batch_data.shape[0]
        elif data_type=='dense':
            x_batch_dom_data, y_batch_data, events_batch_data = next(data_iter)
            x_dom_data = x_batch_dom_data.float().cuda()
            
        #pass together data and MC
        y_dom = y_batch_data.cuda()
        _, out_dom  = net(x_dom_data, alpha)
        out_dom = out_dom.float()
        dom_loss = criterion(out_dom, y_dom)
        loss = clf_loss + dom_loss
        loss.backward()
        optimizer.step()
        y_pred_clf      = out_clf_MC  .argmax(dim=-1)
        y_pred_dom      = out_dom     .argmax(dim=-1)
        accuracy_clf += (y_pred_clf == y_clf_MC).squeeze().sum().item()/len(y_pred_clf)
        accuracy_dom += (y_pred_dom == y_dom).squeeze().sum().item()/len(y_pred_dom)
        epoch_loss_clf += clf_loss.item()
        epoch_loss_dom += dom_loss.item()
        # nb_proc = (i+1)*args.batch_size
        # loop.set_description('Epoch {}/{}'.format(nb_proc, datalen))
        # loop.set_postfix(loss_clf=epoch_loss_clf/(i+1), accuracy_clf=accuracy_clf/(i+1),
        #                  loss_dom_MC=epoch_loss_dom_MC/(i+1), accuracy_dom_MC=accuracy_dom_MC/(i+1),
        #                   accuracy_dom_data=accuracy_dom_data/(i+1), loss_dom_data=epoch_loss_dom_data/(i+1))
    epoch_loss_clf /= (i+1)
    accuracy_clf /= (i+1)
    epoch_loss_dom /= (i+1)
    accuracy_dom /= (i+1)
    print("train  {:5d}: loss_clf:  {:.9f} acc_clf: {:.9f} loss_dom: {:.9f} acc_dom: {:.9f}".format(epoch_id, epoch_loss_clf,
                                                                                                    accuracy_clf, epoch_loss_dom, accuracy_dom))
    clf_loader.sampler.on_epoch_end()
    data_loader.sampler.on_epoch_end()
    return (epoch_loss_clf, accuracy_clf, epoch_loss_dom, accuracy_dom)


def evaluate_valid(epoch_id, net, criterion, clf_loader, data_loader, data_type):
    net.eval()
    print('**********EVAL**************')
    epoch_loss_clf = 0
    accuracy_clf   = 0
    epoch_loss_dom = 0
    accuracy_dom   = 0
    datalen = min(len(clf_loader), len(data_loader))
    loop = range(0, datalen)
    clf_iter = iter(clf_loader)
    data_iter = iter(data_loader)
    alpha = 1.
    for i in loop:
        with torch.autograd.no_grad():
            if data_type=='sparse':
                coordins_batch_MC, features_batch_MC, y_batch_clf_MC, events_batch_MC = next(clf_iter)
                x_clf_MC = coordins_batch_MC, features_batch_MC.cuda(), y_batch_clf_MC.shape[0]
            elif data_type=='dense':
                x_batch_clf_MC, y_batch_clf_MC, events_batch_MC = next(clf_iter)
                x_clf_MC = x_batch_clf_MC.float().cuda()
            y_clf_MC = y_batch_clf_MC.cuda()
            out_clf_MC, out_dom_MC  = net(x_clf_MC, alpha)
            out_clf_MC = out_clf_MC.float()
            
            clf_loss = criterion(out_clf_MC, y_clf_MC)
            
            if data_type=='sparse':
                coordins_batch_data, features_batch_data, y_batch_data, events_batch_data = next(data_iter)
                x_dom_data = coordins_batch_data, features_batch_data.cuda(), y_batch_data.shape[0]
            elif data_type=='dense':
                x_batch_dom_data, y_batch_data, events_batch_data = next(data_iter)
                x_dom_data = x_batch_dom_data.float().cuda()
            #pass together data and MC
            y_dom = y_batch_data.cuda()
            _, out_dom  = net(x_dom_data, alpha)
            out_dom = out_dom.float()
            dom_loss = criterion(out_dom, y_dom)
            y_pred_clf      = out_clf_MC  .argmax(dim=-1)
            y_pred_dom      = out_dom     .argmax(dim=-1)
            #print(y_pred_clf, y_clf_MC, '\n', y_pred_dom_MC, y_dom_MC, '\n', y_pred_dom_data, y_dom_data)
            accuracy_clf += (y_pred_clf == y_clf_MC).squeeze().sum().item()/len(y_pred_clf)
            accuracy_dom += (y_pred_dom == y_dom).squeeze().sum().item()/len(y_pred_dom)
            epoch_loss_clf += clf_loss.item()
            epoch_loss_dom += dom_loss.item()
    epoch_loss_clf /= (i+1)
    accuracy_clf /= (i+1)
    epoch_loss_dom /= (i+1)
    accuracy_dom /= (i+1)
    print("test  {:5d}: loss_clf:  {:.9f} acc_clf: {:.9f} loss_dom: {:.9f} acc_dom: {:.9f}".format(epoch_id, epoch_loss_clf,
                                                                                                    accuracy_clf, epoch_loss_dom, accuracy_dom))
    print('*********************')

    return (epoch_loss_clf, accuracy_clf, epoch_loss_dom, accuracy_dom)


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
          valid_data_df,
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
    datagen_data       = DataGen(data_df, data_type, q=q, augmentation=augmentation,  node='lowTh')
    datagen_data_valid  = DataGen(valid_data_df, data_type, q=q, augmentation=False, node='lowTh')
    
    sampler_clf        = BalancedSampler(datagen_clf)
    sampler_clf_valid  = BalancedSampler(datagen_clf_valid)
    sampler_data       = SimpleSampler(datagen_data)
    sampler_data_valid = SimpleSampler(datagen_data_valid)

    clf_loader = torch.utils.data.DataLoader(datagen_clf, sampler=sampler_clf, batch_size=batch_size, 
                                             shuffle=False, num_workers=num_workers, collate_fn=collate_fn(data_type), drop_last=True, pin_memory=False)
        
    clf_valid_loader = torch.utils.data.DataLoader(datagen_clf_valid, sampler=sampler_clf_valid, batch_size=batch_size, 
                                                   shuffle=False, num_workers=0, collate_fn=collate_fn(data_type), drop_last=True, pin_memory=False)

    data_loader = torch.utils.data.DataLoader(datagen_data, sampler=sampler_data, batch_size=batch_size, 
                                             shuffle=False, num_workers=num_workers, collate_fn=collate_fn(data_type), drop_last=True, pin_memory=False)
        
    data_valid_loader = torch.utils.data.DataLoader(datagen_data_valid, sampler=sampler_data_valid, batch_size=batch_size, 
                                                   shuffle=False, num_workers=0, collate_fn=collate_fn(data_type), drop_last=True, pin_memory=False)
    for i in range(0, nb_epoch):
        train_stats = train_one_epoch(i, 
                                      net,
                                      criterion,
                                      optimizer,
                                      clf_loader, 
                                      data_loader, 
                                      data_type, 
                                      nb_epoch)
        evaluate_stats = evaluate_valid(i, net, criterion, clf_valid_loader, data_valid_loader, data_type)
        writer.add_scalar('Loss_clf/train', train_stats[0], i)
        writer.add_scalar('Loss_clf/test', evaluate_stats[0], i)
        writer.add_scalar('Accuracy_clf/train', train_stats[1], i)
        writer.add_scalar('Accuracy_clf/test', evaluate_stats[1], i)
        writer.add_scalar('Loss_dom/train', train_stats[2], i)
        writer.add_scalar('Loss_dom/test', evaluate_stats[2], i)
        writer.add_scalar('Accuracy_dom/train', train_stats[3], i)
        writer.add_scalar('Accuracy_dom/test', evaluate_stats[3], i)
        if evaluate_stats[0] < save_loss:
            best_loss, best_acc, _, _ = evaluate_stats
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
        q = 0,
        alpha = 1):
    net.eval()
    datagen_test = DataGen(test_df, q=q)
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
            out = nn.functional.softmax(net(x_clf, alpha)[0].float(), dim=1)[:,1]
            y_pred = out.cpu().detach().numpy()
            prediction[idx : idx+len(y_pred)] = y_pred
            evs[idx : idx+len(y_pred)] = events_batch
            idx+=len(y_pred)
    return prediction, evs

