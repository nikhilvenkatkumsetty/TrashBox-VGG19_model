'''                       
Resize script adapted from Thung and Yang's trashnet (https://github.com/garythung/trashnet)              
Accepts images from an input folder, resizes them to dimensions specified in trashnet constants.py,   
and outputs them to a destination folder in subfolders by class.
'''

import os
import resize_constants
import numpy as np
from scipy import misc, ndimage
import imageio
from skimage.transform import resize

def resize_(image, dim1, dim2):
    return resize(image, (dim1, dim2))


def fileWalk(directory, destPath):
    try:
        os.makedirs(destPath)
    except OSError:
        if not os.path.isdir(destPath):
            raise

    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if len(file) <= 4 or file[-4:] != '.jpg':
                continue

            pic = imageio.imread(os.path.join(subdir, file))
            dim1 = len(pic)
            dim2 = len(pic[0])
            if dim1 > dim2:
                pic = np.rot90(pic)
                
            picResized = resize_(pic, resize_constants.DIM1, resize_constants.DIM2)
            imageio.imwrite(os.path.join(destPath, file), picResized) 
            

def main():

    parentPath = os.path.dirname(os.getcwd())

    prepath = os.path.join(parentPath, 'datasets/trashnet/data/dataset-original')
    glassDir = os.path.join(prepath, 'glass')
    paperDir = os.path.join(prepath, 'paper')
    cardboardDir = os.path.join(prepath, 'cardboard')
    plasticDir = os.path.join(prepath, 'plastic')
    metalDir = os.path.join(prepath, 'metal')
    trashDir = os.path.join(prepath, 'trash')
    compostDir = os.path.join(prepath, 'compost')

    destPath = os.path.join(parentPath, 'datasets/trashnet/data/dataset-resized')

    try:
        os.makedirs(destPath)

    except OSError:
        if not os.path.isdir(destPath):
            raise


    #GLASS
    fileWalk(glassDir, os.path.join(destPath, 'glass'))

    #PAPER
    fileWalk(paperDir, os.path.join(destPath, 'paper'))
            
    #CARDBOARD
    fileWalk(cardboardDir, os.path.join(destPath, 'cardboard'))
            
    #PLASTIC
    fileWalk(plasticDir, os.path.join(destPath, 'plastic'))
            
    #METAL
    fileWalk(metalDir, os.path.join(destPath, 'metal'))

    #TRASH
    fileWalk(trashDir, os.path.join(destPath, 'trash'))

    #COMPOST
    fileWalk(compostDir, os.path.join(destPath, 'compost'))

            
if __name__ == '__main__':
    main()
