import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd

def imshow(inp, title=None):
    ''' Imshow for Tensor. '''
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
        
        

def show_databatch(dataloaders, dataset_sizes, class_names):
    ''' Shows a minibatch of data with labels specified in class_names '''
    inputs, classes = next(iter(dataloaders['train']))  
    out = torchvision.utils.make_grid(inputs)
    imshow(out, title=[class_names[x] for x in classes])
                

def create_grid_for_mb(i, inputs, num_images, class_names, preds, labels, file_path_base):
    ''' Creates a grid showing predicted and ground truth labels for subset of images of a minibatch.
        Params:
             -  i:               the  minibatch number 
             -  inputs:          images
             -  num_images:      number of images to plot in the grid; height and width of grid are np.sqrt(num_images)
             -  class_names:     class labels
             -  preds:           model predictions 
             -  labels:          ground truth labels
             -  file_path_base:  base of file path to save the created image
    '''
    
    plt.clf()
    fig = plt.figure(figsize=(15, 15))
    images_so_far = 0
    
    for j in range(inputs.size()[0]):
        images_so_far += 1
        ax = fig.add_subplot(np.sqrt(num_images), np.sqrt(num_images), images_so_far)
        plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.3)
        ax.axis('off')
        ax.set_title('predicted: {}\n actual: {}'.format(class_names[preds[j]], class_names[labels[j]]), fontsize = 'medium')
        imshow(inputs.cpu().data[j])
        plt.show()
        
        if images_so_far >= num_images:
            plt.savefig(file_path_base + "predictions_" + str(i) + ".png")
            break        

    
def show_confusion_mat(matrix, num_classes, class_names, outfile=None):
    ''' Displays confusion matrix '''

    fig = plt.figure(1) 
    ax = plt.subplot()
    sn.heatmap(matrix, annot=True, annot_kws={"size": 10}, ax = ax, fmt='g') 

    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels');
    ax.set_title('Confusion Matrix')
    ax.xaxis.set_ticklabels(class_names, fontsize=7)
    ax.yaxis.set_ticklabels(class_names, fontsize=7)
    
    plt.show()
    if outfile != None:
        plt.savefig(outfile)
    
