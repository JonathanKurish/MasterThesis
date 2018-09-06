from random import randint, uniform
import numpy as np
import cv2
import scipy.ndimage as scipy
from math import floor

# Selects Augmentation Function Based On Number of Streams
def augment_data(X, param_dict, phase):
    num_streams = len(param_dict['channels'].split(","))
    if num_streams>1:
        return augment_two_stream_data(X, param_dict, phase)
    else:
        return augment_single_stream_data(X, param_dict, phase)

# Perform Augmentation On Batch of Data In Single Stream Network
def augment_single_stream_data(X, param_dict, phase):
    network_type = param_dict['network_type']
    augment_data = param_dict['augment_data']
    batch_size = len(X)
    channels = int(param_dict['channels'])

    augmented_data = np.zeros((batch_size,224,224,channels))
    for idx,x in enumerate(X):
        if augment_data == "True" and phase=="train":
            if network_type == "optical_flow":
                x = do_flow_augmentation(x)
            else:
                x = do_spatial_augmentation(x)
        x = subtract_mean(x)
                
        x = random_flip(x)
        x = resize_img(x)
        augmented_data[idx] = x


    return augmented_data

# Perform Augmentation On Batch of Data In Two Stream Network
def augment_two_stream_data(X, param_dict, phase):
    network_type = param_dict['network_type']
    augment_data = param_dict['augment_data']
    channels1,channels2 = [int(c) for c in param_dict['channels'].split(",")]

    X1s, X2s = X
    batch_size = len(X1s)

    augmented_stream1 = np.zeros((batch_size,224,224,channels1))
    augmented_stream2 = np.zeros((batch_size,224,224,channels2))

    for idx,x1 in enumerate(X1s):
        x2 = X2s[idx]


        if network_type=="multiplier":
            xdim, ydim, zdim = np.array(x1).shape
            x1_new = np.zeros((xdim, ydim+1, zdim))
            x1_new[:,0:ydim,:] = x1
            x1 = x1_new


        both = np.concatenate((x1, x2), axis=2)
        if augment_data=="True" and phase=="train":
            both = do_spatial_augmentation(both)

        both = subtract_mean(both)
        both = random_flip(both)
        both = resize_img(both)

        augmented_x1 = both[...,0:channels1]
        augmented_x2 = both[...,channels1:]

        augmented_stream1[idx] = augmented_x1
        augmented_stream2[idx] = augmented_x2

    return [augmented_stream1, augmented_stream2]


def in_temporal_stream(network_type):
    return network_type in ["optical_flow","temporal", "single_grey_diff", "multi_grey", "multi_grey_diff"]

def in_spatial_stream(network_type):
    return network_type in ['spatial', 'single_grey']

def in_both_streams(network_type):
    return network_type in ["multiplier",
                            "multi_grey_single_rgb",
                            "multi_and_single_grey",
                            "multi_diff_and_single_grey",
                            "multi_grey_diff_single_rgb"]

# Subtracts Mean From Images. Only Applied On Spatial Input
def subtract_mean(img):
    xdim,ydim,channels = img.shape
    res = np.zeros((xdim,ydim,channels))
    for i in range(0,channels):
        mean = np.mean(img[:,:,i])
        res[:,:,i] = img[:,:,i] - mean
    return res


# Augmentations Used In Article
def do_spatial_augmentation(img):
    img = random_RGB_zoom(img)
    img = random_crop(img)
    return img

# Resizes Img To 224x224xChannels
def resize_img(img):
    xdim,ydim,channels = img.shape
    resized_img = np.zeros((224,224,channels))
    if channels == 1:
        resized_img[:,:,0] = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    else:
        resized_img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    return resized_img

# Flow Augmentations Used In Article
def do_flow_augmentation(imgs):
        sizes = [256, 224, 192, 168]
        new_sizeX = sizes[randint(0,3)]
        new_sizeY = sizes[randint(0,3)]

        width, height = np.array((imgs[:,:,0])).shape

        if (width < new_sizeX):
            x_start = 0
            new_sizeX = width
        else:
            x_start = randint(0,width-new_sizeX)
        if (height < new_sizeY):
            y_start = 0
            new_sizeY = height
        else:
            y_start = randint(0,height-new_sizeY)

        new_img = imgs[x_start:(x_start+new_sizeX),y_start:(y_start+new_sizeY)]
        return new_img

def random_flip(input):
    flip = randint(0,1)
    if (flip == 1):
        return np.fliplr(input)
    return input

# Creates a zoomed version of the input with +- 25 % at random for each axis
def random_RGB_zoom(img):
    zoomX = uniform(0.75,1.25)
    zoomY = uniform(0.75,1.25)
    zoomX = float(zoomX)
    zoomY = float(zoomY)
    zoomImg = scipy.zoom(img, [zoomX, zoomY, 1])
    return zoomImg

# randomly crops and image (after it has been zoomed)
def random_crop(img):
    
    # define random distance from edges
    random_width_start = uniform(0, 0.25)
    random_height_start = uniform(0, 0.25)
    random_width_end = uniform(0, 0.25)
    random_height_end = uniform(0, 0.25)
    
    # determine cropped image sizes
    height, width, channels = img.shape
    start_pos_x = floor(width * random_width_start)
    start_pos_y = floor(height * random_height_start)
    end_pos_x = floor(width - (width * random_width_end))
    end_pos_y = floor(height - (height * random_height_end))
    
    # crop the zoomed image
    cropped_img = img[start_pos_x:end_pos_x, start_pos_y:end_pos_y,:]
    return cropped_img
