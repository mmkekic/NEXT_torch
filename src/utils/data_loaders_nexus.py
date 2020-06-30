import torch
import numpy as np
import tables as tb

masked_pos = np.array([[145.0, -185.0], [-65.0, -215.0], [-165.0, -115.0], [35.0, -185.0], [35.0, -195.0], [-65.0, 15.0],[25.0, 85.0], [85.0, 125.0], [25.0, -15.0], [-5.0,  -55.0]]) 

def get_mchits(filename, event):
    with tb.open_file(filename) as tab:
        indx_tab = tab.root.MC.extents.get_where_list( 'evt_number == {}'.format(event)) 
        last_hit = tab.root.MC.extents[indx_tab]['last_hit']
        first_hit = tab.root.MC.extents[indx_tab[0]-1]['last_hit'] if indx_tab[0]>0 else -1
        hits_ = tab.root.MC.hits[int(first_hit+1):int(last_hit+1)][['hit_position', 'hit_energy']]
        hits=np.zeros((len(hits_), 4))
        hits[:, :3] = hits_['hit_position']
        hits[:, 3] = hits_['hit_energy']
    return hits
                                                                
def get_3d_input_nexus(file, event, datatype, binsX, binsY, binsZ, *, augmentation):

    chits = get_mchits(file, event)

    if augmentation:
        rand_disp = np.random.randn(*chits.shape)*[5, 5, 5, 1]
        rand_disp[:,3] *= 0.05*chits[:, 3]
        chits = rand_disp + chits
        zmov=30
        maxX, minX = max(chits[:,0]), min(chits[:,0])
        maxY, minY = max(chits[:,1]), min(chits[:,1])
        maxZ, minZ = max(chits[:,2]), min(chits[:,2])
        max_mov_X = min(max(binsX)- maxX, zmov)//10
        min_mov_X = max(min(binsX)- minX, -zmov)//10+1
        max_mov_Y = min(max(binsY)- maxY, zmov)//10
        min_mov_Y = max(min(binsY)- minY, -zmov)//10+1
        max_mov_Z = min(max(binsZ)- maxZ, zmov)//5
        min_mov_Z = max(min(binsZ)- minZ, -zmov)//5+1
        if max_mov_X > min_mov_X:
            chits[:,0] += np.random.randint(min_mov_X, max_mov_X, 1)*10
        if max_mov_Y > min_mov_Y:
            chits[:,1] += np.random.randint(min_mov_Y, max_mov_Y, 1)*10
        if max_mov_Z > min_mov_Z:
            chits[:,2] += np.random.randint(min_mov_Z, max_mov_Z, 1)*5

        flip = np.random.choice([True, False],3)
        Xc = (maxX+minX)/2
        Yc = (maxY+minY)/2
        Zc = (maxZ+minZ)/2
        
        if flip[0]:
            chits[:,0] = 2*Xc-chits[:,0]
        if flip[1]:
            chits[:,1] = 2*Yc-chits[:,1]
        if flip[2]:
            chits[:,2] = 2*Zc-chits[:,2]

        rots = np.random.choice([True, False], 3)
        if rots[0]:
            X_new = minX + chits[:,1]-minY
            Y_new = minY + chits[:,0]-minX
            chits[:,0] = X_new
            chits[:,1] = Y_new

        if rots[1]:
            X_new = minX + chits[:,2]-minZ
            Z_new = minZ + chits[:,0]-minX
            chits[:,0] = X_new
            chits[:,2] = Z_new

        if rots[2]:
            Y_new = minY + chits[:,2]-minZ
            Z_new = minZ + chits[:,1]-minY
            chits[:,1] = Y_new
            chits[:,2] = Z_new

        random_drop = (chits[:,2]<0.02*chits[:,2].mean()) | (np.random.choice([True, False], len(chits)))                                                                                                                     
        chits = chits[random_drop] 
        # zoomX = np.random.uniform(0.4,1.2)
        # zoomY = np.random.uniform(0.4,1.2)
        # zoomZ = np.random.uniform(0.4,2.)

        # chits[:,0] = minX + zoomX * (chits[:,0]-minX)
        # chits[:,1] = minY + zoomY * (chits[:,1]-minY)
        # chits[:,2] = minZ + zoomZ * (chits[:,2]-minZ)
  
    if datatype == 'dense':
        x_vals = np.histogramdd(np.concatenate([chits[:,0][:,None], chits[:,1][:,None], chits[:,2][:,None]],                                            
                                               axis =-1), bins=(binsX, binsY, binsZ), weights = np.nan_to_num(chits[:,3]))
        if x_vals[0].sum() ==0:
            return x_vals[0][None, :].astype(np.float32)
        else:
            return (x_vals[0]/x_vals[0].sum())[None, :].astype(np.float32)


    
    elif datatype == 'sparse':
        x_vals = np.digitize(chits[:,0], binsX)
        y_vals = np.digitize(chits[:,1], binsY)
        z_vals = np.digitize(chits[:,2], binsZ)
        evals  = np.nan_to_num(chits[:,3])
        mask   = (x_vals>=0) & (x_vals<len(binsX)) & (y_vals>=0) & (y_vals<len(binsY)) & (z_vals>=0) & (z_vals<len(binsZ)) & (evals>0)
        evals  = evals/sum(evals)
        mask = mask & (evals>1e-6)
        return x_vals[mask].copy(), y_vals[mask].copy(), z_vals[mask].copy(), evals[mask].copy()

class DataGen_nexus(torch.utils.data.Dataset):
    def __init__(self, df, datatype, augmentation=False, **kwargs):
        self.df = df
        self.binsX = np.linspace(-200, 200,  41)
        self.binsY = np.linspace(-200, 200,  41)
        self.binsZ = np.linspace(   0, 550, 111)
        self.augmentation = augmentation
        self.datatype = datatype
    def __getitem__(self, idx):
        file, event = self.df.iloc[idx][['filename', 'event']]
        try:
            y = self.df.iloc[idx].label
        except AttributeError:
            y = -1

        if self.datatype == 'dense':
            x = get_3d_input_nexus(file, event, 'dense', self.binsX, self.binsY, self.binsZ, augmentation=self.augmentation)
            return x, float(y), int(event)
        elif self.datatype == 'sparse':
            xs, ys, zs, es = get_3d_input_nexus (file, event,  'sparse', self.binsX, self.binsY, self.binsZ, augmentation=self.augmentation)
        return xs, ys, zs, es, float(y), int(event)

    def __len__(self):
        return len(self.df)
