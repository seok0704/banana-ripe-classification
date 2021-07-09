import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
from torchvision import datasets

def constructed_model(model_transfer, hidden_layer_size):
    transfer_learning_dict = {'ResNet152': models.resnet152(pretrained=True), 
                              'ResNet101': models.resnet101(pretrained=True),
                              'InceptionV3': models.inception_v3(pretrained=True),
                             }
    
    
    model_transfer = transfer_learning_dict[model_transfer]

    #Remove the last layer from the model
    classifier_name, old_classifier = model_transfer._modules.popitem()
    
    #Freeze the parameters to prevent back-propagation
    for param in model_transfer.parameters():
        param.requires_grad = False
    
    classifier_input_size = old_classifier.in_features

    classifier = nn.Sequential(OrderedDict([
                               ('fc1', nn.Linear(classifier_input_size, hidden_layer_size)),
                               ('relu', nn.ReLU()),
                               ('dropout', nn.Dropout(p=0.25)),
                               ('fc2', nn.Linear(hidden_layer_size, 3)),
                               ]))
    
    model_transfer.add_module(classifier_name, classifier)
    
    return model_transfer