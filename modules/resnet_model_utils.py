from keras.optimizers import SGD
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, add
from keras.layers import Dropout, Flatten, Reshape, Activation
from keras.layers import LeakyReLU, multiply, Lambda, Conv3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import numpy as np

import resnet_models as rm

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def late_pool(x, pooling, late_pool_name):
    if pooling=="average":
        x = AveragePooling2D((7, 7), name=late_pool_name)(x)
        print("Average Pooling applied late")
    elif pooling=="max":
        x = MaxPooling2D((7,7), name=late_pool_name)(x)
        print("Max Pooling applied late")
    elif pooling=="none":
        print("No pooling applied late")
    return x

# Returns The Activation Function Passed (ReLU and Leaky ReLU are only ones used)
def make_activation(x, name):
    af = rm.af
    if (af == 'leaky_relu'):
        x = LeakyReLU(alpha=0.3, name=name)(x)
    else:
        x = Activation(af, name=name)(x)
    return x

####### Functions for weight scaling
def scale_by_class_freq(class_weights, x):
    arguments={"class_weights":class_weights}
    myCustomLayer = Lambda(function=matrix_mul, arguments=arguments)
    return myCustomLayer(x)

def matrix_mul(x, class_weights):
    class_weights_var = K.variable(value=class_weights, dtype='float32')
    return multiply([class_weights_var,x])

def the_reshaping(input_shape):
    return input_shape
#######

# Updates the weight naming prefix by the passed amount of steps.
# Used to more easily interpret weights in the .h5 file using hdf5_viewer
def update_prefix(num=1):
    prefix = rm.prefix
    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    second_in_prefix = prefix[1]
    index_2 = letters.index(second_in_prefix)
    if (index_2+num) > 25:
        first = letters[letters.index(prefix[0])+1]
        second = letters[index_2+num-26]
    else:
        first = prefix[0]
        second = letters[index_2+num]
    
    rm.prefix = first+second+"_"

# Reduces the weight naming prefix by the passed amount of steps.
# Used to more easily interpret weights in the .h5 file using hdf5_viewer
def reduce_prefix(num=10):
    prefix = rm.prefix

    letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    index_2 = letters.index(prefix[1])
    if index_2 >= num:
        first = prefix[0]
        second = letters[index_2-num]
    else:
        rest = num - index_2
        first = letters[letters.index(prefix[0])-1]
        second = letters[25-rest+1]
    
    rm.prefix = first+second+"_"


# Identity Block In The Original ResNet Model
# Only Naming Convention Is Altered
def identity_block(input_tensor, kernel_size, filters, base_name="", stream_str=None):
    if stream_str==None:
        stream = rm.stream
    else:
        stream = stream_str
    
    prefix = rm.prefix
    bn_axis = rm.bn_axis
    

    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters

    # PART A
    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter1, (1, 1), name=prefix+base_name+'cnv_a'+stream)(input_tensor)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+base_name+'bn_a'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+"rel_a"+stream)

    # PART B
    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=prefix+base_name+'cnv_b'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+base_name+'bn_b'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+'rel_b'+stream)

    # PART C
    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter3, (1, 1), name=prefix+base_name+'cnv_c'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+base_name+'bn_c'+stream)(x)

    update_prefix()
    prefix = rm.prefix
    x = add([x, input_tensor], name=prefix+base_name+'res'+stream)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+'rel_c'+stream)
    return x

# Convolutional Block In The Original ResNet Model
# Only Naming Convention Is Altered
def conv_block(input_tensor, kernel_size, filters, base_name="",strides=(2, 2), stream_str=None):
    if stream_str==None:
        stream = rm.stream
    else:
        stream = stream_str
    
    prefix = rm.prefix
    bn_axis = rm.bn_axis

    
    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters


    # PART A
    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter1, (1, 1), strides=strides,name=prefix+base_name+'cnv_a'+stream)(input_tensor)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+base_name+'bn_a'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+'rel_a'+stream)

    # PART B
    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=prefix+base_name+'cnv_b'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+base_name+'bn_b'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+'rel_b'+stream)

    # PART C
    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter3, (1, 1), name=prefix+base_name+'cnv_c'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+base_name+'bn_c'+stream)(x)

    # WHY THIS CONVOLUTION?!
    update_prefix()
    prefix = rm.prefix
    shortcut = Conv2D(nb_filter3, (1, 1), strides=strides,
                             name=prefix+base_name+'cnv_sc'+stream)(input_tensor)
    update_prefix()
    prefix = rm.prefix
    shortcut = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+base_name+'bn_sc'+stream)(shortcut)

    update_prefix()
    prefix = rm.prefix
    x = add([x, shortcut], name=prefix+base_name+'res'+stream)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+'rel_c'+stream)
    return x


def injection_block(input_skip, input_tensor, kernel_size, filters, base_name="",strides=(2, 2), stream_str=None):
    if stream_str==None:
        stream = rm.stream
    else:
        stream = stream_str

    prefix = rm.prefix
    bn_axis = rm.bn_axis

    eps = 1.1e-5
    nb_filter1, nb_filter2, nb_filter3 = filters

    # PART A
    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter1, (1, 1), name=prefix+base_name+'cnv_a'+stream)(input_tensor)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(axis=bn_axis, name=prefix+base_name+'bn_a'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+'rel_a'+stream)

    # PART B
    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=prefix+base_name+'cnv_b'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+base_name+'bn_b'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+'rel_b'+stream)

    # Injection part
    shape = K.int_shape(input_tensor)[1]
    x = Reshape((shape,shape,nb_filter1,1), name=prefix+'rshp1'+stream)(x)
    x = Conv3D(1, (1, 1, 3), padding='same', kernel_initializer=injection_kernel_init, name=prefix+'tmp_inj'+stream)(x)
    x = MaxPooling3D(pool_size=(1, 1, 3), strides=(1,1,1),padding='same', name=prefix+"mx_pl"+stream)(x)
    x = Reshape((shape,shape,nb_filter1), name=prefix+'rhsp2'+stream)(x)
    # No longer injection part

    update_prefix()
    prefix = rm.prefix
    x = Conv2D(nb_filter3, (1, 1), name=prefix+base_name+'cnv_c'+stream)(x)
    update_prefix()
    prefix = rm.prefix
    x = BatchNormalization(axis=bn_axis, name=prefix+base_name+'bn_c'+stream)(x)


    update_prefix()
    prefix = rm.prefix
    x = add([x, input_skip], name=prefix+base_name+'res'+stream)
    update_prefix()
    prefix = rm.prefix
    x = make_activation(x, name=prefix+base_name+'rel_c'+stream)
    return x


def injection_kernel_init(shape, dtype=None):
    ker = np.zeros(shape, dtype=dtype)
    ker[0][0][1]=1
    return ker



def conv2_bot(depth, x, kernel_size, filters):
    if depth==20:
        update_prefix(10)
    else:
        x = identity_block(x, kernel_size, filters, base_name='2c_')
    return x

def conv3_bot(depth, x, kernel_size, filters):
    if depth==50:
        x = identity_block(x, kernel_size, filters, base_name='3c1_')
        x = identity_block(x, kernel_size, filters, base_name='3c2_')
    elif depth==38:
        x = identity_block(x, kernel_size, filters, base_name='3c1_')
        update_prefix(10)
    elif depth==26:
        update_prefix(10)
        update_prefix(10)
    elif depth==20:
        update_prefix(10)
        update_prefix(10)

    return x

def conv4_bot(depth, x, kernel_size, filters):
    if depth==50:
        x = identity_block(x, kernel_size, filters, base_name='4c1_')
        x = identity_block(x, kernel_size, filters, base_name='4c2_')
        x = identity_block(x, kernel_size, filters, base_name='4c3_')
        x = identity_block(x, kernel_size, filters, base_name='4c4_')
    elif depth==38:
        x = identity_block(x, kernel_size, filters, base_name='4c1_')
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)
    elif depth==26 or depth==20:
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)

    return x



def conv5(depth, x, kernel_size, filters):
    if depth==50 or depth==38:
        x = conv_block(x, kernel_size, filters, base_name='5a_')
        x = identity_block(x, kernel_size, filters, base_name='5b_')
        x = identity_block(x, kernel_size, filters, base_name='5c_')
    elif depth==26:
        x = conv_block(x, kernel_size, filters, base_name='5a_')
        x = identity_block(x, kernel_size, filters, base_name='5b_')
        update_prefix(10)
    elif depth==20:
        update_prefix(12)
        update_prefix(10)
        update_prefix(10)
    return x


def conv2_bot_multiplier(depths, x1, x2, kernel_size, filters):
    depth1 = depths[0]
    depth2 = depths[1]
    if depth1==20:
        update_prefix(10)
    else:
        x1 = identity_block(x1, kernel_size, filters, base_name='2c_', stream_str='_1')
    
    reduce_prefix(10)
    
    if depth2==20:
        update_prefix(10)
    else:
        x2 = identity_block(x2, kernel_size, filters, base_name='2c_', stream_str='_2')
    return x1, x2

def conv3_bot_multiplier(depths, x1, x2, kernel_size, filters):
    depth1 = depths[0]
    depth2 = depths[1]

    if depth1==50:
        x1 = identity_block(x1, kernel_size, filters, base_name='3c1_', stream_str='_1')
        x1 = identity_block(x1, kernel_size, filters, base_name='3c2_', stream_str='_1')
    elif depth1==38:
        x1 = identity_block(x1, kernel_size, filters, base_name='3c1_', stream_str='_1')
        update_prefix(10)
    elif depth1==26 or depth1==20:
        update_prefix(10)
        update_prefix(10)

    reduce_prefix(10)
    reduce_prefix(10)
    if depth2==50:
        x2 = identity_block(x2, kernel_size, filters, base_name='3c1_', stream_str='_2')
        x2 = identity_block(x2, kernel_size, filters, base_name='3c2_', stream_str='_2')
    elif depth2==38:
        x2 = identity_block(x2, kernel_size, filters, base_name='3c1_', stream_str='_2')
        update_prefix(10)
    elif depth2==26 or depth2==20:
        update_prefix(10)
        update_prefix(10)

    return x1, x2


def conv4_bot_multiplier(depths, x1, x2, kernel_size, filters):
    depth1 = depths[0]
    depth2 = depths[1]

    if depth1==50:
        x1 = identity_block(x1, kernel_size, filters, base_name='4c1_', stream_str='_1')
        x1 = identity_block(x1, kernel_size, filters, base_name='4c2_', stream_str='_1')
        x1 = identity_block(x1, kernel_size, filters, base_name='4c3_', stream_str='_1')
        x1 = identity_block(x1, kernel_size, filters, base_name='4c4_', stream_str='_1')
    elif depth1==38:
        x1 = identity_block(x1, kernel_size, filters, base_name='4c1_', stream_str='_1')
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)
    elif depth1==26 or depth1==20:
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)
        
    reduce_prefix(10)
    reduce_prefix(10)
    reduce_prefix(10)
    reduce_prefix(10)

    if depth2==50:
        x2 = identity_block(x2, kernel_size, filters, base_name='4c1_', stream_str='_2')
        x2 = identity_block(x2, kernel_size, filters, base_name='4c2_', stream_str='_2')
        x2 = identity_block(x2, kernel_size, filters, base_name='4c3_', stream_str='_2')
        x2 = identity_block(x2, kernel_size, filters, base_name='4c4_', stream_str='_2')
    
    elif depth2==38:
        x2 = identity_block(x2, kernel_size, filters, base_name='4c1_', stream_str='_2')
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)
    elif depth2==26 or depth2==20:
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)
        update_prefix(10)

    return x1, x2

def conv5_multiplier(depths, x1, x2, kernel_size, filters):
    depth1 = depths[0]
    depth2 = depths[1]

    if depth1==50 or depth1==38:
        x1 = conv_block(x1, kernel_size, filters, base_name='5a_',stream_str='_1')
        x1 = identity_block(x1, kernel_size, filters, base_name='5b_',stream_str='_1')
        x1 = identity_block(x1, kernel_size, filters, base_name='5c_',stream_str='_1')
    elif depth1==26:
        x1 = conv_block(x1, kernel_size, filters, base_name='5a_',stream_str='_1')
        x1 = identity_block(x1, kernel_size, filters, base_name='5b_',stream_str='_1')
        update_prefix(10)
    elif depth1==20:
        update_prefix(12)
        update_prefix(10)
        update_prefix(10)

        
    reduce_prefix(12)
    reduce_prefix(10)
    reduce_prefix(10)



    if depth2==50 or depth2==38:
        x2 = conv_block(x2, kernel_size, filters, base_name='5a_',stream_str='_2')
        x2 = identity_block(x2, kernel_size, filters, base_name='5b_',stream_str='_2')
        x2 = identity_block(x2, kernel_size, filters, base_name='5c_',stream_str='_2')

    elif depth2==26:
        x2 = conv_block(x2, kernel_size, filters, base_name='5a_',stream_str='_2')
        x2 = identity_block(x2, kernel_size, filters, base_name='5b_',stream_str='_2')
        update_prefix(10)

    elif depth2==20:
        update_prefix(12)
        update_prefix(10)
        update_prefix(10)

    return x1, x2
