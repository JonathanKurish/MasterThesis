from keras.layers import Input, Dense, Conv2D, MaxPooling2D, AveragePooling2D, add
from keras.layers import Dropout, Flatten, Activation, multiply
from keras.layers.normalization import BatchNormalization
from keras.models import Model

from keras.utils import plot_model

import resnet_model_utils as rmu

def single_stream_resnet(param_dict, stream_str, class_distribution):
    global af
    global stream
    global bn_axis
    global prefix

    input_shape=(224,224,int(param_dict['channels']))
    depth=int(param_dict['depth'])
    scale_before_softmax=param_dict['scale_weights_on_input']
    af = param_dict['activation_function']
    pooling = param_dict['pooling']
    dropout = float(param_dict['dropout'])
    num_classes = int(param_dict['num_classes'])
    if 'rename_dense' in param_dict:
        rename_dense = rmu.str2bool(param_dict['rename_dense'])
    else:
        rename_dense = False
    stream = stream_str
    bn_axis = 3
    prefix = 'aa_'
    eps = 1.1e-5

    img_input = Input(shape=(input_shape), name=prefix+"input"+stream)
    #Conv1
    rmu.update_prefix()
    x = Conv2D(64, (7, 7), strides=(2, 2), name=prefix+'cnv1'+stream, padding='same')(img_input)
    rmu.update_prefix()
    x = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+'bn1'+stream)(x)
    rmu.update_prefix()
    x = rmu.make_activation(x, prefix+'rel1'+stream)
    rmu.update_prefix()
    x = MaxPooling2D((3, 3), strides=(2, 2), name=prefix+'pool1'+stream)(x)

    #Conv2 Top
    x = rmu.conv_block(x, 3, [64, 64, 256], base_name='2a_',strides=(1, 1))
    #Conv2 Mid
    x = rmu.identity_block(x, 3, [64, 64, 256], base_name='2b_')
    #Conv2 Bot
    x = rmu.conv2_bot(depth, x, 3, [64, 64, 256])


    #Conv3 Top
    x = rmu.conv_block(x, 3, [128, 128, 512], base_name='3a_')
    #Conv3 Mid
    x = rmu.identity_block(x, 3, [128, 128, 512], base_name='3b_')
    #Conv3 Bot
    x = rmu.conv3_bot(depth, x, 3, [128, 128, 512])


    #Conv4 Top
    x = rmu.conv_block(x, 3, [256, 256, 1024], base_name='4a_')
    #Conv4 Mid 
    x = rmu.identity_block(x, 3, [256, 256, 1024], base_name='4b_')
    #Conv4 Bot
    x = rmu.conv4_bot(depth, x, 3, [256, 256, 1024])


    #Full Conv5
    x = rmu.conv5(depth, x, 3, [512, 512, 2048])

    #Late Pooling
    rmu.update_prefix()
    late_pool_name = prefix+'pool' + stream
    x = rmu.late_pool(x, pooling, late_pool_name)

    #Late Flattening
    rmu.update_prefix()
    x = Flatten(name=prefix+'flat'+stream)(x)

    #Added extra dropout as per article
    rmu.update_prefix()
    x = Dropout(rate=dropout, name=prefix+'drop'+stream)(x)


    #Trim down to #num_classes nodes, one for each class
    rmu.update_prefix()
    if rename_dense:
        dense_name = prefix + "fc_renamed" + stream
    else:
        dense_name = prefix + "fc" + stream
    x = Dense(num_classes, name=dense_name)(x)
    
    #Here we add weights based on input frequency if chosen
    if rmu.str2bool(scale_before_softmax):
        print("using class_distribution")
        x = rmu.scale_by_class_freq(class_distribution, x)
        
    #Softmax layer
    rmu.update_prefix()
    x = Activation('softmax', name=prefix+'smax'+stream)(x)



    model = Model(img_input, x, name='resnet'+stream)

    return model



def multiplier_resnet(param_dict, class_distribution):
    global af
    global bn_axis
    global prefix
    af = param_dict['activation_function']
    stream1='_1'
    stream2='_2'
    bn_axis = 3
    prefix = 'aa_'
    eps = 1.1e-5
    channels = param_dict['channels'].split(",")
    channels1 = int(channels[0])
    channels2 = int(channels[1])
    input_shape1 = (224,224,channels1)
    input_shape2 = (224,224,channels2)
    dropout = float(param_dict['dropout'])
    pooling = param_dict['pooling']
    depths = [int(d) for d in param_dict['depth'].split(",")]
    scale_before_softmax=param_dict['scale_weights_on_input']
    num_classes = int(param_dict['num_classes'])
    rename_dense = rmu.str2bool(param_dict['rename_dense'])
    depth1 = depths[0]
    depth2 = depths[1]
    
    img_input1 = Input(shape=(input_shape1), name=prefix+"input"+stream1)
    img_input2 = Input(shape=(input_shape2), name=prefix+"input"+stream2)
    
    #Conv1
    rmu.update_prefix()
    x_temporal = Conv2D(64, (7, 7), strides=(2, 2), name=prefix+'cnv1'+stream1, padding='same')(img_input1)
    x_spatial = Conv2D(64, (7, 7), strides=(2, 2), name=prefix+'cnv1'+stream2, padding='same')(img_input2)

    rmu.update_prefix()
    x_temporal = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+'bn1'+stream1)(x_temporal)
    x_spatial = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+'bn1'+stream2)(x_spatial)


    rmu.update_prefix()
    x_temporal = Activation('relu', name=prefix+'rel1'+stream1)(x_temporal)
    x_spatial = Activation('relu', name=prefix+'rel1'+stream2)(x_spatial)

    #Pool1
    rmu.update_prefix()
    x_temporal = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=prefix+'pool1'+stream1)(x_temporal)
    x_spatial = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=prefix+'pool1'+stream2)(x_spatial)


    #Conv2 Top
    x_temporal = rmu.conv_block(x_temporal, 3, [64, 64, 256], base_name='2a_', strides=(1, 1), stream_str=stream1)
    rmu.reduce_prefix(12)
    x_spatial = rmu.conv_block(x_spatial, 3, [64, 64, 256], base_name='2a_', strides=(1, 1), stream_str=stream2)

    # Multiplicative gate from motion into apperearance (temporal -> spatial)
    spatial_fused = multiply([x_spatial, x_temporal], name=prefix+'2a_mulgate')
    print("Temporal Stream Shape: ", x_temporal._keras_shape)
    print("Spatial Stream Shape: ", x_spatial._keras_shape)

    #Conv2 Mid
    x_temporal = rmu.identity_block(x_temporal, 3, [64, 64, 256], base_name='2b_', stream_str=stream1)
    rmu.reduce_prefix(10)
    x_spatial = rmu.injection_block(x_spatial, spatial_fused, 3, [64, 64, 256], base_name='2b_', stream_str=stream2)
    
    #Conv2 Bot
    x_temporal, x_spatial = rmu.conv2_bot_multiplier(depths, x_temporal, x_spatial, 3, [64,64,256])

    #Conv3 Top
    #x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',t=t,bias=bias1,rel=rel)
    x_temporal = rmu.conv_block(x_temporal, 3, [128, 128, 512], base_name='3a_', stream_str=stream1)
    rmu.reduce_prefix(12)
    x_spatial = rmu.conv_block(x_spatial, 3, [128, 128, 512], base_name='3a_', stream_str=stream2)

    # Multiplicative gate from motion into apperearance (temporal -> spatial)
    spatial_fused = multiply([x_spatial, x_temporal], name=prefix+'3a_mulgate')

    #Conv3 Mid
    x_temporal = rmu.identity_block(x_temporal, 3, [128, 128, 512], base_name='3b_', stream_str=stream1)
    rmu.reduce_prefix(10)
    x_spatial = rmu.injection_block(x_spatial, spatial_fused, 3, [128, 128, 512], base_name='3b_', stream_str=stream2)
    
    #Conv3 Bot
    x_temporal, x_spatial = rmu.conv3_bot_multiplier(depths, x_temporal, x_spatial, 3, [128, 128, 512])

    #Conv4 top
    x_temporal = rmu.conv_block(x_temporal, 3, [256, 256, 1024], base_name='4a_', stream_str=stream1)
    rmu.reduce_prefix(12)
    x_spatial = rmu.conv_block(x_spatial, 3, [256, 256, 1024], base_name='4a_', stream_str=stream2)

    # Multiplicative gate from motion into apperearance (temporal -> spatial)
    spatial_fused = multiply([x_spatial, x_temporal], name=prefix+'4a_mulgate')

    # Conv4 Mid
    x_temporal = rmu.identity_block(x_temporal, 3, [256, 256, 1024], base_name='4b_', stream_str=stream1)
    rmu.reduce_prefix(10)
    #x_temporal = conv_block(x_temporal, 3, [256, 256, 1024], base_name='4b1_', af=af, stream=stream1)
    x_spatial = rmu.injection_block(x_spatial, spatial_fused, 3, [256, 256, 1024], base_name='4b_', stream_str=stream2)
    
    # Conv4 Bot
    x_temporal, x_spatial = rmu.conv4_bot_multiplier(depths, x_temporal, x_spatial, 3, [256, 256, 1024])
    
    #Conv5 Full
    x_temporal, x_spatial = rmu.conv5_multiplier(depths, x_temporal, x_spatial, 3, [512, 512, 2048])

    #Pool5
    #x_fc = AveragePooling2D((7, 7), name='avg_pool'+t)(x)
    rmu.update_prefix()
    x_temporal = AveragePooling2D((7, 7), name=prefix+'pool'+stream1, padding='same')(x_temporal)
    x_spatial = AveragePooling2D((7, 7), name=prefix+'pool'+stream2, padding='same')(x_spatial)

    # Flatten data
    rmu.update_prefix()
    x_temporal = Flatten(name=prefix+'flat'+stream1)(x_temporal)
    x_spatial = Flatten(name=prefix+'flat'+stream2)(x_spatial)

    #Trim down to #num_classes nodes, one for each class
    rmu.update_prefix()
    rmu.update_prefix()
    x_temporal = Dense(num_classes, name=prefix+'fc'+stream1)(x_temporal)
    x_spatial = Dense(num_classes, name=prefix+'fc'+stream2)(x_spatial)
    
    #Here we add weights based on input frequency if chosen
    if rmu.str2bool(scale_before_softmax):
        x_temporal = rmu.scale_by_class_freq(class_distribution, x_temporal)
        x_spatial = rmu.scale_by_class_freq(class_distribution, x_spatial)
        
    #Softmax layer
    rmu.update_prefix()
    #x_temporal = Activation('softmax', name=prefix+'smax'+stream1)(x_temporal)
    #x_spatial = Activation('softmax', name=prefix+'smax'+stream2)(x_spatial)


    rmu.update_prefix()
    x_together = add([x_temporal, x_spatial], name=prefix+'fusion')
    #x_together = add([x_temporal, x_spatial], name='output_name')
    x_together = Activation('softmax', name='main_output')(x_together)
    rmu.update_prefix()
    #x_together = Dropout(rate=dropout, name=prefix+'dropout')(x_together)
    #rmu.update_prefix()
    #if rename_dense:
    #    output_name = "main_outputX"
    #else:
    #    output_name = "main_output"
    #x_together = Dense(num_classes, activation='softmax', name=output_name)(x_together)

    model = Model(inputs=[img_input1, img_input2], outputs=[x_together])


    #print("Saving model to file")
    #plot_model(model, to_file='XM_multiplier_model.png',show_shapes=True)

    return model


def late_fusion_two_stream_resnet(param_dict, class_distribution):
    global af
    global bn_axis
    global prefix
    af = param_dict['activation_function']
    stream1='_1'
    stream2='_2'
    bn_axis = 3
    prefix = 'aa_'
    eps = 1.1e-5
    channels = param_dict['channels'].split(",")
    channels1 = int(channels[0])
    channels2 = int(channels[1])
    input_shape1 = (224,224,channels1)
    input_shape2 = (224,224,channels2)
    dropout = float(param_dict['dropout'])
    pooling = param_dict['pooling']
    depths = [int(d) for d in param_dict['depth'].split(",")]
    scale_before_softmax=param_dict['scale_weights_on_input']
    num_classes = int(param_dict['num_classes'])
    rename_dense = rmu.str2bool(param_dict['rename_dense'])
    depth1 = depths[0]
    depth2 = depths[1]
    
    img_input1 = Input(shape=(input_shape1), name=prefix+"input"+stream1)
    img_input2 = Input(shape=(input_shape2), name=prefix+"input"+stream2)
    
    #Conv1
    rmu.update_prefix()
    x_temporal = Conv2D(64, (7, 7), strides=(2, 2), name=prefix+'cnv1'+stream1, padding='same')(img_input1)
    x_spatial = Conv2D(64, (7, 7), strides=(2, 2), name=prefix+'cnv1'+stream2, padding='same')(img_input2)

    rmu.update_prefix()
    x_temporal = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+'bn1'+stream1)(x_temporal)
    x_spatial = BatchNormalization(epsilon=eps, axis=bn_axis, name=prefix+'bn1'+stream2)(x_spatial)


    rmu.update_prefix()
    x_temporal = Activation('relu', name=prefix+'rel1'+stream1)(x_temporal)
    x_spatial = Activation('relu', name=prefix+'rel1'+stream2)(x_spatial)

    #Pool1
    rmu.update_prefix()
    x_temporal = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=prefix+'pool1'+stream1)(x_temporal)
    x_spatial = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name=prefix+'pool1'+stream2)(x_spatial)


    #Conv2 Top
    x_temporal = rmu.conv_block(x_temporal, 3, [64, 64, 256], base_name='2a_', strides=(1, 1), stream_str=stream1)
    rmu.reduce_prefix(12)
    x_spatial = rmu.conv_block(x_spatial, 3, [64, 64, 256], base_name='2a_', strides=(1, 1), stream_str=stream2)

    #Conv2 Mid
    x_temporal = rmu.identity_block(x_temporal, 3, [64, 64, 256], base_name='2b_', stream_str=stream1)
    rmu.reduce_prefix(10)
    x_spatial = rmu.identity_block(x_spatial, 3, [64, 64, 256], base_name='2b_', stream_str=stream2)
    
    #Conv2 Bot
    x_temporal, x_spatial = rmu.conv2_bot_multiplier(depths, x_temporal, x_spatial, 3, [64,64,256])

    #Conv3 Top
    #x = conv_block(x, 3, [128, 128, 512], stage=3, block='a',t=t,bias=bias1,rel=rel)
    x_temporal = rmu.conv_block(x_temporal, 3, [128, 128, 512], base_name='3a_', stream_str=stream1)
    rmu.reduce_prefix(12)
    x_spatial = rmu.conv_block(x_spatial, 3, [128, 128, 512], base_name='3a_', stream_str=stream2)

    #Conv3 Mid
    x_temporal = rmu.identity_block(x_temporal, 3, [128, 128, 512], base_name='3b_', stream_str=stream1)
    rmu.reduce_prefix(10)
    x_spatial = rmu.identity_block(x_spatial, 3, [128, 128, 512], base_name='3b_', stream_str=stream2)
    
    #Conv3 Bot
    x_temporal, x_spatial = rmu.conv3_bot_multiplier(depths, x_temporal, x_spatial, 3, [128, 128, 512])

    #Conv4 top
    x_temporal = rmu.conv_block(x_temporal, 3, [256, 256, 1024], base_name='4a_', stream_str=stream1)
    rmu.reduce_prefix(12)
    x_spatial = rmu.conv_block(x_spatial, 3, [256, 256, 1024], base_name='4a_', stream_str=stream2)

    # Conv4 Mid
    x_temporal = rmu.identity_block(x_temporal, 3, [256, 256, 1024], base_name='4b_', stream_str=stream1)
    rmu.reduce_prefix(10)
    x_spatial = rmu.identity_block(x_spatial, 3, [256, 256, 1024], base_name='4b_', stream_str=stream2)
    
    # Conv4 Bot
    x_temporal, x_spatial = rmu.conv4_bot_multiplier(depths, x_temporal, x_spatial, 3, [256, 256, 1024])
    
    #Conv5 Full
    x_temporal, x_spatial = rmu.conv5_multiplier(depths, x_temporal, x_spatial, 3, [512, 512, 2048])

    #Pool5
    #x_fc = AveragePooling2D((7, 7), name='avg_pool'+t)(x)
    rmu.update_prefix()
    x_temporal = AveragePooling2D((7, 7), name=prefix+'pool'+stream1, padding='same')(x_temporal)
    x_spatial = AveragePooling2D((7, 7), name=prefix+'pool'+stream2, padding='same')(x_spatial)

    # Flatten data
    rmu.update_prefix()
    x_temporal = Flatten(name=prefix+'flat'+stream1)(x_temporal)
    x_spatial = Flatten(name=prefix+'flat'+stream2)(x_spatial)

    #Trim down to #num_classes nodes, one for each class
    rmu.update_prefix()
    rmu.update_prefix()
    x_temporal = Dense(num_classes, name=prefix+'fc'+stream1)(x_temporal)
    x_spatial = Dense(num_classes, name=prefix+'fc'+stream2)(x_spatial)
    
    #Here we add weights based on input frequency if chosen
    if rmu.str2bool(scale_before_softmax):
        x_temporal = rmu.scale_by_class_freq(class_distribution, x_temporal)
        x_spatial = rmu.scale_by_class_freq(class_distribution, x_spatial)
        
    #Softmax layer
    rmu.update_prefix()
    #x_temporal = Activation('softmax', name=prefix+'smax'+stream1)(x_temporal)
    #x_spatial = Activation('softmax', name=prefix+'smax'+stream2)(x_spatial)


    rmu.update_prefix()
    x_together = add([x_temporal, x_spatial], name=prefix+'fusion')
    #x_together = add([x_temporal, x_spatial], name='output_name')
    x_together = Activation('softmax', name='main_output')(x_together)
    rmu.update_prefix()
    #x_together = Dropout(rate=dropout, name=prefix+'dropout')(x_together)
    #rmu.update_prefix()
    #if rename_dense:
    #    output_name = "main_outputX"
    #else:
    #    output_name = "main_output"
    #x_together = Dense(num_classes, activation='softmax', name=output_name)(x_together)

    model = Model(inputs=[img_input1, img_input2], outputs=[x_together])


    #print("Saving model to file")
    #plot_model(model, to_file='XM_multiplier_model.png',show_shapes=True)

    return model