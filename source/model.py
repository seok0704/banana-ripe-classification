import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from torchvision import datasets

def constructed_model(model_transfer, hidden_layer_size):
    
    #Remove the last layer from the model
    classifier_name, old_classifier = model._modules.popitem()
    
    #Freeze the parameters to prevent back-propagation
    for param in model_transfer.parameters():
        param.requires_grad = False
    
    classifier_input_size = old_classifier.in_features

    classifier = nn.Sequential(OrderedDict([
                               ('fc1', nn.Linear(classifier_input_size, hidden_layer_size)),
                               ('activation', nn.Relu()),
                               ('dropout', nn.Dropout(p=0.3)),
                               ('fc2', nn.Linear(hidden_layer_size, 3)),
                               ('output', nn.LogSoftmax(dim=1))
                               ]))
    model.add_module(classifier_name, classifier)
    
    return model