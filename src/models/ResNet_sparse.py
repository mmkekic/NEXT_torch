import sparseconvnet as scn
from torch.autograd import Function
import torch

def SparseResNet(dimension, nInputPlanes, layers, mom=0.99):
    """
    pre-activated ResNet
    e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
    """
    nPlanes = nInputPlanes
    m = scn.Sequential()

    def residual(nIn, nOut, stride):
        if stride > 1:
            return scn.Convolution(dimension, nIn, nOut, 3, stride, False)
        elif nIn != nOut:
            return scn.NetworkInNetwork(nIn, nOut, False)
        else:
            return scn.Identity()
    for blockType, n, reps, stride in layers:
        for rep in range(reps):
            if blockType[0] == 'b':  # basic block
                if rep == 0:
                    m.add(scn.BatchNormReLU(nPlanes, momentum=mom, eps=1e-5))
                    m.add(
                        scn.ConcatTable().add(
                            scn.Sequential().add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False) if stride == 1 else scn.Convolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    stride,
                                    False)) .add(
                                scn.BatchNormReLU(n, momentum=mom, eps=1e-5)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            residual(
                                nPlanes,
                                n,
                                stride)))
                else:
                    m.add(
                        scn.ConcatTable().add(
                            scn.Sequential().add(
                                scn.BatchNormReLU(nPlanes, momentum=mom, eps=1e-5)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False)) .add(
                                scn.BatchNormReLU(n, momentum=mom, eps=1e-5)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            scn.Identity()))
            nPlanes = n
            m.add(scn.AddTable())
    m.add(scn.BatchNormReLU(nPlanes, momentum=mom, eps=1e-5))
    return m



class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class Feature_extr(torch.nn.Module):
    def __init__(self, spatial_size=(99, 99, 199), n_initial_filters=8, mom=0.99, ):
        torch.nn.Module.__init__(self)
        self.input_tensor = scn.InputLayer(dimension=3, spatial_size= (spatial_size))
        self.initial_convolution1 = scn.SubmanifoldConvolution(
                dimension   = 3,
                nIn         = 1,
                nOut        = n_initial_filters,
                filter_size = (7, 7, 15),
                bias        = False
            )
        self.relu1 = scn.BatchNormReLU(n_initial_filters, momentum=mom, eps=1e-5)
        self.initial_convolution2 = scn.SubmanifoldConvolution(
                dimension   = 3,
                nIn         = n_initial_filters,
                nOut        = n_initial_filters,
                filter_size = (7, 7, 15),
                bias        = False
            )
        self.relu2 = scn.BatchNormReLU(n_initial_filters, momentum=mom, eps=1e-5)
        n_filters = 2*n_initial_filters
        self.initial_downsample = scn.Convolution(
                dimension   = 3,
                nIn         = n_initial_filters,
                nOut        = n_filters,
                filter_size = (7, 7, 15),
                filter_stride = (2, 2, 4),
                bias        = False
            )
        
        self.resnet_block = SparseResNet(3, n_filters,
                                         [['b', n_filters   , 1, 1],
                                          ['b', 2*n_filters , 1, 2],
                                          ['b', 2*n_filters , 1, 1],
                                          ['b', 4*n_filters , 1, 2],
                                          ['b', 4*n_filters , 1, 1],
                                          ['b', 8*n_filters , 1, 2],
                                          ['b', 8*n_filters , 1, 1],
                                          ['b', 16*n_filters, 1, 2]], 
                                         mom = mom)
        n_filters=16*n_filters
        sz_pool_fin = 2
        self.n_final_filters = n_filters
        self.pool = scn.AveragePooling(3, sz_pool_fin, 1)
        self.sparse_to_dense= scn.SparseToDense(dimension=3, nPlanes=n_filters)        

    def forward(self, x):
        x = self.input_tensor(x) 
        x = self.initial_convolution1(x)
        x = self.relu1(x)
        x = self.initial_convolution2(x)
        x = self.relu2(x)
        x = self.initial_downsample(x)
        x = self.resnet_block(x)
        x = self.pool(x)
        x = self.sparse_to_dense(x)
        x = x.view(x.size(0), -1)
        return x




def SparseResNet_noBN(dimension, nInputPlanes, layers, mom=0.99):
    """
    pre-activated ResNet
    e.g. layers = {{'basic',16,2,1},{'basic',32,2}}
    """
    nPlanes = nInputPlanes
    m = scn.Sequential()

    def residual(nIn, nOut, stride):
        if stride > 1:
            return scn.Convolution(dimension, nIn, nOut, 3, stride, False)
        elif nIn != nOut:
            return scn.NetworkInNetwork(nIn, nOut, False)
        else:
            return scn.Identity()
    for blockType, n, reps, stride in layers:
        for rep in range(reps):
            if blockType[0] == 'b':  # basic block
                if rep == 0:
                    m.add(
                        scn.ConcatTable().add(
                            scn.Sequential().add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False) if stride == 1 else scn.Convolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    stride,
                                    False)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            residual(
                                nPlanes,
                                n,
                                stride)))
                else:
                    m.add(
                        scn.ConcatTable().add(
                            scn.Sequential() .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    nPlanes,
                                    n,
                                    3,
                                    False)) .add(
                                scn.SubmanifoldConvolution(
                                    dimension,
                                    n,
                                    n,
                                    3,
                                    False))) .add(
                            scn.Identity()))
            nPlanes = n
            m.add(scn.AddTable())
    return m

