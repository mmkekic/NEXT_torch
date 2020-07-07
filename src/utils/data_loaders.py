import torch
import numpy as np
import tables as tb

masked_pos = np.array([[145.0, -185.0], [-65.0, -215.0], [-165.0, -115.0], [35.0, -185.0], [35.0, -195.0], [-65.0, 15.0],[25.0, 85.0], [85.0, 125.0], [25.0, -15.0], [-5.0,  -55.0]]) 

def get_3d_input(file, event, datatype, binsX, binsY, binsZ, *, group, node, q, augmentation, mean, std):
    if augmentation:
        minq = max(6, q-10)
        maxq = min(30, q+10)
        q = np.random.uniform(minq, maxq)
    with tb.open_file(file) as tab:
        chits = tab.root[group][node].read_where( '(event == {}) & (Q >= {})'.format(event, q))
    #centralize image
    # X_oft = 0 - (max(chits['X'])+min(chits['X']))/2.
    # Y_oft = 0 - (max(chits['Y'])+min(chits['Y']))/2.
    # Z_oft = 250. - (max(chits['Z'])+min(chits['Z']))/2.
    # chits['X'] += X_oft
    # chits['Y'] += Y_oft
    # chits['Z'] += Z_oft
    if augmentation:
        zmov = 50
        maxX, minX = max(chits['X']), min(chits['X'])
        maxY, minY = max(chits['Y']), min(chits['Y'])
        maxZ, minZ = max(chits['Z']), min(chits['Z'])
        max_mov_X = min(max(binsX)- maxX, zmov)//10
        min_mov_X = max(min(binsX)- minX, -zmov)//10+1
        max_mov_Y = min(max(binsY)- maxY, zmov)//10
        min_mov_Y = max(min(binsY)- minY, -zmov)//10+1
        max_mov_Z = min(max(binsZ)- maxZ, zmov)//5
        min_mov_Z = max(min(binsZ)- minZ, -zmov)//5+1
        if max_mov_X > min_mov_X:
            chits['X'] += np.random.randint(min_mov_X, max_mov_X, 1)*10
        if max_mov_Y > min_mov_Y:
            chits['Y'] += np.random.randint(min_mov_Y, max_mov_Y, 1)*10
        if max_mov_Z > min_mov_Z:
            chits['Z'] += np.random.randint(min_mov_Z, max_mov_Z, 1)*5

        flip = np.random.choice([True, False],3)
        Xc = (maxX+minX)/2
        Yc = (maxY+minY)/2
        Zc = (maxZ+minZ)/2
        if flip[0]:
            chits['X'] = 2*Xc-chits['X']
        if flip[1]:
            chits['Y'] = 2*Yc-chits['Y']
        # if flip[2]:
        #     chits['Z'] = 2*Zc-chits['Z']

        # rot_xy = np.random.choice([True, False])
        # if rot_xy:
        #     chits[['Y','X']] = chits[['X', 'Y']]
                    
        
        rot_xy = np.random.choice([True, False])
        if rot_xy:
            # X_new = chits['Y']
            # Y_new = chits['X']
            X_new = minX + chits['Y']-minY
            Y_new = minY + chits['X']-minX
            chits['X'] = X_new
            chits['Y'] = Y_new

        # zoomX = np.random.uniform(0.8, 1.5)
        # zoomY = np.random.uniform(0.8, 1.5)
        # zoomZ = np.random.uniform(0.8, 1.5)
        # # zoomX = np.random.normal(loc=1.05, scale=0.1)
        # # zoomY = np.random.normal(loc=1.05, scale=0.1)
        # # zoomZ = np.random.normal(loc=1.05, scale=0.1)

        # chits['X'] = minX + zoomX * (chits['X']-minX)
        # chits['Y'] = minY + zoomY * (chits['Y']-minY)
        # chits['Z'] = minZ + zoomZ * (chits['Z']-minZ)
  
    if datatype == 'dense':
        x_vals = np.histogramdd(np.concatenate([chits['X'][:,None], chits['Y'][:,None], chits['Z'][:,None]],                                            
                                               axis =-1), bins=(binsX, binsY, binsZ), weights = np.nan_to_num(chits['Ec']))
        if x_vals[0].sum() ==0:
            return x_vals[0][None, :].astype(np.float32)
        else:
            if mean is not None:
                return ((x_vals[0]/x_vals[0].sum())[None, :].astype(np.float32)-mean)/std
            else:
                return (x_vals[0]/x_vals[0].sum())[None, :].astype(np.float32)

    
    elif datatype == 'sparse':
        x_vals = np.digitize(chits['X'], binsX)
        y_vals = np.digitize(chits['Y'], binsY)
        z_vals = np.digitize(chits['Z'], binsZ)
        evals  = np.nan_to_num(chits['Ec'])
        mask   = (x_vals>=0) & (x_vals<len(binsX)) & (y_vals>=0) & (y_vals<len(binsY)) & (z_vals>=0) & (z_vals<len(binsZ)) & (evals>0)
        evals  = evals/sum(evals)
        mask = mask & (evals>1e-6)
        if mean is not None:
            return x_vals[mask].copy(), y_vals[mask].copy(), z_vals[mask].copy(), (evals[mask].copy()-mean)/std
        else:
            return x_vals[mask].copy(), y_vals[mask].copy(), z_vals[mask].copy(), evals[mask].copy()

class DataGen(torch.utils.data.Dataset):
    def __init__(self, df, datatype, augmentation=False, group='CHITS', node='lowTh', q=0, mean=None, std=None):
        self.df = df
        self.binsX = np.linspace(-200, 200,  41)
        self.binsY = np.linspace(-200, 200,  41)
        self.binsZ = np.linspace(   0, 550, 111)
        self.group = group
        self.node  = node
        self.q     = q
        self.augmentation = augmentation
        self.datatype = datatype
        self.mean = mean
        self.std = std
    def __getitem__(self, idx):
        file, event = self.df.iloc[idx][['filename', 'event']]
        try:
            y = self.df.iloc[idx].label
        except AttributeError:
            y = -1

        if self.datatype == 'dense':
            x = get_3d_input(file, event, 'dense', self.binsX, self.binsY, self.binsZ, q=self.q, group=self.group, node=self.node, augmentation=self.augmentation, mean=self.mean, std=self.std)
            return x, float(y), int(event)
        elif self.datatype == 'sparse':
            xs, ys, zs, es = get_3d_input (file, event,  'sparse', self.binsX, self.binsY, self.binsZ, q=self.q, group=self.group, node=self.node, augmentation=self.augmentation, mean=self.mean, std=self.std)
        return xs, ys, zs, es, float(y), int(event)

    def __len__(self):
        return len(self.df)

class BalancedSampler(torch.utils.data.Sampler):
    def __init__ (self, dataset):
        self.dataset = dataset
        self.pos_labels = self.dataset.df[self.dataset.df.label == 1].index.values
        self.neg_labels = self.dataset.df[self.dataset.df.label == 0].index.values

    def __iter__(self):
        for i in range(int(self.__len__()/2)):
            yield self.neg_labels[i]
            yield self.pos_labels[i]

    def __len__(self):
        return 2*min(len(self.pos_labels), len(self.neg_labels))

    def on_epoch_end(self):
        np.random.shuffle(self.pos_labels)
        np.random.shuffle(self.neg_labels)
        
class SimpleSampler(torch.utils.data.Sampler):
    def __init__ (self, dataset):
        self.dataset = dataset
        self.data_indx = self.dataset.df.index.values

    def __iter__(self):
        for i in range(int(self.__len__())):
            yield self.data_indx[i]

    def __len__(self):
        return len(self.data_indx)

    def on_epoch_end(self):
        np.random.shuffle(self.data_indx)

def merge(batch):
    vals   = []
    coords = []
    labels = torch.zeros(len(batch)).int()
    events = np.zeros(len(batch))
    for bi, data in enumerate(batch):
        xs, ys, zs, es, y, event = data
        bs = torch.ones(len(xs)).long()*bi
        coords.append(torch.cat([torch.from_numpy(xs).unsqueeze(-1),
                                 torch.from_numpy(ys).unsqueeze(-1),
                                 torch.from_numpy(zs).unsqueeze(-1),
                                 bs.unsqueeze(-1)], axis=1))
        vals.append(torch.from_numpy(es).unsqueeze(-1))
        labels[bi] = int(y)
        events[bi] = event
    coords = torch.cat(coords, axis=0)
    vals   = torch.cat(vals, axis=0)
    y_batch        = labels.long()
    features_batch = vals.float()
    coordins_batch = coords.long()
    return coordins_batch, features_batch, y_batch, events

def merge_dense(batch):
    labels = torch.zeros(len(batch)).long()
    events = np.zeros(len(batch))
    inputs = torch.zeros(len(batch), *batch[0][0].shape).float()
    for bi, data in enumerate(batch):
        x, y, event = data
        inputs[bi] = torch.as_tensor(x, dtype=torch.float32)
        labels[bi] = int(y)
        events[bi] = event

    y_batch = labels.long()
    x_batch = inputs.float()
    return x_batch, y_batch, events


def collate_fn(datatype):
    if datatype=='dense':
        return merge_dense
    elif datatype=='sparse':
        return merge
