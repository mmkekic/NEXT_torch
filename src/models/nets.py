import torch
import torch.nn as nn
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

