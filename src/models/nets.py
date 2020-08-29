import torch
import torch.nn as nn
from torch.autograd import Function

class Classifier(torch.nn.Module):
    def __init__(self, feature_extr, n_filters): #either sparse or dense
        torch.nn.Module.__init__(self)
        self.feature_extr = feature_extr
        self.dropout = nn.Dropout()
        self.class_classifier = nn.Sequential()
        lin_c1 = nn.Linear(n_filters, 32, bias=True)
        self.class_classifier.add_module('c_fc1', lin_c1)
        self.class_classifier.add_module('c_relu1', nn.ReLU())
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        lin_c2 = nn.Linear(32, 2, bias=True)
        self.class_classifier.add_module('c_fc3', lin_c2)

    def forward(self, x):
        x = self.feature_extr(x)
        output = self.dropout(x)
        class_output = self.class_classifier(output)
        return class_output


def SparseClassifier(mom):
    from . import ResNet_sparse as rn   
    features_extr = rn.Feature_extr(mom=mom)
    n_final_filters = features_extr.n_final_filters
    model = Classifier(features_extr, n_final_filters)
    return model


def DenseClassifier(mom):
    from . import ResNet_dense as rn
    features_extr =  rn.Feature_extr(mom=mom)
    n_final_filters = features_extr.n_final_filters
    model = Classifier(features_extr, n_final_filters)
    return model


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DANN(torch.nn.Module):
    def __init__(self, feature_extr, n_filters): #either sparse or dense
        torch.nn.Module.__init__(self)
        self.feature_extr = feature_extr
        self.dropout = nn.Dropout()
        self.class_classifier = nn.Sequential()
        lin_c1 = nn.Linear(n_filters, 32, bias=True)
        self.class_classifier.add_module('c_fc1', lin_c1)
        self.class_classifier.add_module('c_relu1', nn.ReLU())
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        lin_c2 = nn.Linear(32, 2, bias=True)
        self.class_classifier.add_module('c_fc3', lin_c2)        
        self.domain_classifier = nn.Sequential()
        lin_d1 = nn.Linear(n_filters, 128, bias=True)
        self.domain_classifier.add_module('d_fc1', lin_d1)
        self.domain_classifier.add_module('d_relu1', nn.ReLU())
        self.domain_classifier.add_module('d_drop1', nn.Dropout())
        lin_d2 = nn.Linear(128, 32, bias=True)
        self.domain_classifier.add_module('d_fc2', lin_d2)
        self.domain_classifier.add_module('d_relu2', nn.ReLU())
        lin_d3 = nn.Linear(32, 2, bias=True)
        #torch.nn.init.xavier_normal_(lin_d3.weight)
        self.domain_classifier.add_module('d_fc3', lin_d3)
        #self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

    def forward(self, x, alpha):
        x = self.feature_extr(x)
        output = self.dropout(x)
        reverse_feature = ReverseLayerF.apply(output, alpha)
        class_output = self.class_classifier(output)
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output


def SparseDANN(mom):
    from . import ResNet_sparse as rn   
    features_extr = rn.Feature_extr(mom=mom)
    n_final_filters = features_extr.n_final_filters
    model = DANN(features_extr, n_final_filters)
    return model


def DenseDANN(mom):
    from . import ResNet_dense as rn
    features_extr =  rn.Feature_extr(mom=mom)
    n_final_filters = features_extr.n_final_filters
    model = DANN(features_extr, n_final_filters)
    return model


class Unique_Classifier(torch.nn.Module):
    def __init__(self, n_filters): #either sparse or dense
        torch.nn.Module.__init__(self)
        self.dropout = nn.Dropout()
        self.class_classifier = nn.Sequential()
        lin_c1 = nn.Linear(n_filters, 32, bias=True)
        self.class_classifier.add_module('c_fc1', lin_c1)
        self.class_classifier.add_module('c_relu1', nn.ReLU())
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        lin_c2 = nn.Linear(32, 2, bias=True)
        self.class_classifier.add_module('c_fc3', lin_c2)

    def forward(self, x):
        output = self.dropout(x)
        class_output = self.class_classifier(output)
        return class_output

