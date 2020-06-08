import os
import yaml
from argparse     import ArgumentParser
from src.utils import train_utils as tu
from src.utils import train_utils_dann as tud
import pandas as pd
import torch
import torch.nn as nn

def is_valid_conf_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        with open(arg, 'r') as stream:
            try:
                return yaml.safe_load(stream)
            except yaml.YAMLError:
                parser.error("The file %s not valid configuration!" % arg)
def is_valid_action(parser, arg):
    if not arg in ['train', 'predict']:
        parser.error("The action %s is not allowed!" % arg)
    else:
        return arg
DataTypes = ['sparse', 'dense']
NetTypes = ['classifier', 'dann']
dataset = '/lustre/ific.uv.es/ml/ific020/DANN.h5'

def train_classifier(model, data_type, lr, n_epochs, batch_size, folder_name, q):
    train_df  = pd.read_hdf(dataset, key='MC_train').sample(frac=1.).reset_index(drop=True)
    valid_df  = pd.read_hdf(dataset, key='MC_valid').sample(frac=1.).reset_index(drop=True)
    sig_ratio = train_df.label.sum()/len(train_df)
    weights = torch.tensor([sig_ratio, 1.-sig_ratio],device='cuda').float()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    criterion = nn.CrossEntropyLoss(weights)
    model_path = folder_name
    os.makedirs(model_path)
    os.makedirs(model_path+'/logs')
    tu.train(net=model, data_type=data_type, criterion=criterion, optimizer=optimizer, 
             scheduler=scheduler, batch_size=batch_size, nb_epoch=n_epochs, 
             train_df=train_df, valid_df=valid_df, q=q, num_workers=4, model_path=folder_name, augmentation=True)


def train_dann(model, data_type, lr, n_epochs, batch_size, folder_name, q):
    train_df  = pd.read_hdf(dataset, key='MC_train').sample(frac=1.).reset_index(drop=True)
    valid_df  = pd.read_hdf(dataset, key='MC_valid').sample(frac=1.).reset_index(drop=True)
    data_df  = pd.read_hdf(dataset, key='data_train').sample(frac=1.).reset_index(drop=True)
    valid_data_df  = pd.read_hdf(dataset, key='data_valid').sample(frac=1.).reset_index(drop=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')
    criterion = nn.CrossEntropyLoss()
    model_path = folder_name
    os.makedirs(model_path)
    os.makedirs(model_path+'/logs')
    tud.train(net=model, data_type=data_type, criterion=criterion, optimizer=optimizer, 
             scheduler=scheduler, batch_size=batch_size, nb_epoch=n_epochs, 
             train_df=train_df, valid_df=valid_df, data_df=data_df, valid_data_df=valid_data_df,
              q=q, num_workers=4, model_path=folder_name, augmentation=True)

# MC
def pred_MC(model, net_type, datatype, q, num_workers, folder, format_name):
    df_test  = pd.read_hdf(dataset, key='MC_valid')
    df_test_prime = df_test.drop(['label'], axis=1)
    if net_type == 'classifier':
        predict=tu.predict
    elif net_type=='dann':
        predict = tud.predict
    prd = predict(model,
                  df_test_prime,
                  datatype,
                  batch_size=1024,
                  num_workers=num_workers,
                  q=q)
    prd_df = pd.DataFrame({'event':prd[1], 'predicted':prd[0].flatten()})
    df_test=df_test.merge(prd_df, on='event')
    df_test.to_csv(f'{folder}/MC_6206_prediction_{format_name}.csv', index=False)
    print ('MC written')

def pred_data(model, net_type, datatype, q, num_workers, folder, format_name, runs=[7470, 7471, 7472, 7473]):
    df_train  = pd.read_hdf(dataset, key='data_train')
    df_test  = pd.read_hdf(dataset, key='data_test')
    df_data_all = pd.concat([df_train, df_test], ignore_index=True)
    if net_type == 'classifier':
        predict=tu.predict
    elif net_type=='dann':
        predict = tud.predict
    for run in runs:
        df_data = df_data_all[df_data_all.run_number == run]
        prd_data = predict(model,
                           df_data,
                           datatype,
                           batch_size = 1024,
                           num_workers = num_workers, 
                           q=q)
        prd_df = pd.DataFrame({'event':prd_data[1], 'predicted':prd_data[0].flatten()})
        df_data.event= df_data.event.astype('int')
        prd_df.event = prd_df.event.astype('int')
        df_data=df_data.merge(prd_df, on='event')
        df_data.to_csv(f"{folder}/data_{run}_prediction_{format_name}.csv", index=False) 
        print(f'data {run} written')


if __name__ == '__main__':
    parser = ArgumentParser(description="parameters for models")
    parser.add_argument("-conf", dest = "confname", required=True,
                        help = "input file with parameters", metavar="FILE",
                        type = lambda x: (x, is_valid_conf_file(parser, x)))
    parser.add_argument("-a", dest = "action" , required = True,
                        help = "action to do for NN",
                        type = lambda x : is_valid_action(parser, x))
    args   = parser.parse_args()
    conf_param_name = (args.confname[0].split('/')[-1]).split('.')[0]
    params = args.confname[1]
    action = args.action

    data_type = params['data_type']
    net_type = params['net_type']
    if data_type not in DataTypes:
        raise KeyError('data_type not understood')
    #construct model
    if net_type == 'classifier':
        if data_type == 'sparse':
            from src.models.nets import SparseClassifier as clf
        elif data_type == 'dense':
            from src.models.nets import DenseClassifier as clf 
        model = clf(params['mom'])
        model.cuda()

    elif net_type == 'dann':
        if data_type == 'sparse':
            from src.models.nets import SparseDANN as dann
        elif data_type == 'dense':
            from src.models.nets import DenseDANN as dann
        model = dann(params['mom'])
        model.cuda()


    weights = params['saved_weights']
    if weights:
        model.load_state_dict(torch.load(weights)['state_dict'])

    if action == 'predict':
        folder = params['predict_folder']
        format_name = params['predict_name']
        pred_MC(model, net_type, data_type, 4, folder, format_name)
        pred_data(model, net_type, data_type, 4, folder, format_name, runs=[7470, 7471, 7472, 7473])
    
    elif action == 'train':
        if net_type == 'classifier':
            folder_name = params['train_folder']
            train_classifier(model, data_type, params['lr'], params['n_epochs'], params['batch_size'], folder_name, params['q_cut'])
        elif net_type == 'dann':
            folder_name = params['train_folder']
            train_dann(model, data_type, params['lr'], params['n_epochs'], params['batch_size'], folder_name, params['q_cut'])
