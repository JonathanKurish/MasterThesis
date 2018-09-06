from keras import optimizers
import data_augmentation as da
from random import shuffle
import numpy as np
import get_data as gd
from os.path import isfile

from vgg_models import VGG_16, VGG_6
from resnet_models import multiplier_resnet, single_stream_resnet
from resnet_models import late_fusion_two_stream_resnet

# Converts String To Bool
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

# Select What Network Model To Construct
def make_network_model(param_dict, stream_str, class_distribution=None):
    network_type = param_dict['network_type']
    if da.in_both_streams(network_type):
        return multiplier_resnet(param_dict,class_distribution)
    elif network_type == "late_fusion_two_stream_resnet":
    	return late_fusion_two_stream_resnet(param_dict,class_distribution)
    elif network_type == 'vgg16':
        return VGG_16()
    elif network_type == 'vgg6':
        return VGG_6()
    #elif network_type == 'mobilenet':
    #    return mobilenet_model()
    else:
        return single_stream_resnet(param_dict, stream_str,class_distribution)

# Returns The Training Set Class Distribution, Potentially Applied In Training Phase
def get_class_distribution(Y):
	x,y = Y.shape
	class_occurences = np.zeros((1,y))
	for elem in Y:
		the_class = elem.argmax()
		class_occurences[:, the_class] += 1

	avg = x/y
	class_distribution = 1 / (class_occurences / avg)

	return class_distribution

# Helper Function To Check Number of Layers That Were Changed When Loading In Weights
def check_weights(pre, after, names):
	num_didnt_change = sum([1 for (layer1,layer2) in zip(pre, after)
		if np.array_equal(np.array(layer1),np.array(layer2))])
	print("Number of parts that didnt change:",num_didnt_change)



# Loads In Pre-Trained Weights, Whether It Be Imagenet Or Our Own Results
def load_weights(model, param_dict, stream, by_name=False, phase="train"):
    initial_weights = param_dict['initial_weights'].split(",")
    channels =param_dict['channels'].split(",")
    reload_weights = str2bool(param_dict['reload_weights'])
    savefile_name = param_dict['savefile_name']

    base = 'h5_weights/imagenet50_'
    base2 = 'channels_stream'

    if stream=="both":
    	stream="_1"

    if reload_weights and phase=="train":
        # Load weights from previous version of this model
        if isfile("h5_weights/"+savefile_name+".h5"):
                model.load_weights("h5_weights/"+savefile_name+".h5", by_name=True)
                print("Loaded In Weights From Previous run")
        else:
        	print("File for reload weights doesnt exist")

    elif "rgb_imagenet" in initial_weights[0]:
        print("Loading imagenet RGB weights now")
        model.load_weights("h5_weights/imagenet50_20channels_rgb_stream1.h5", by_name=True)
    else:
        for idx, weights in enumerate(initial_weights):
            if weights[0]=='i':
            	network_type = param_dict['network_type']
            	if "vgg" in network_type:
            		print("Loading vgg imagenet weights for", network_type)
            		model.load_weights("h5_weights/"+network_type+"_imagenet_weights.h5", by_name=True)
            	#elif "mobilenet" in network_type:
            	#	print("Loading mobilenet imagenet weights for", network_type)
            	#	model.load_weights("h5_weights/mobilenet_imagenet_weights.h5")

            	else:
            		weights_name = base + str(channels[idx]) + base2 + stream[1] + '.h5'
            		print("Loading weights:", weights_name)
            		model.load_weights(weights_name, by_name = by_name)
            elif param_dict['network_type'] == "vgg16":
            	model.load_weights()

            elif weights != "none":
                weights_name = "h5_weights/" + weights + ".h5"
                print("Loading weights:", weights_name)
                model.load_weights(weights_name, by_name = by_name)
            stream="_2"

    return model

# Shuffles The Dataset
def shuffle_data(X,Y):
  combined = list(zip(X, Y))
  shuffle(combined)
  X_shuffled, Y_shuffled = zip(*combined)
  return np.array(X_shuffled), np.array(Y_shuffled)

# Prepares Parameters For Model Construction
def prepare_stream(network_type):
    if da.in_temporal_stream(network_type):
        stream='_1'
        input_names = ['aa_input_1']
        output_name = 'gv_smax' + stream
    elif da.in_spatial_stream(network_type):
        stream='_2'
        input_names = ['aa_input_2']
        output_name = 'gv_smax' + stream
    elif da.in_both_streams(network_type):
        stream='both'
        input_names = ['aa_input_1', 'aa_input_2']
        output_name = 'main_output'
    elif network_type == "late_fusion_two_stream_resnet":
        stream='both'
        input_names = ['aa_input_1', 'aa_input_2']
        output_name = 'main_output'

    elif "vgg" in network_type:
    	stream=""
    	input_names = ['block1_conv1_input']
    	output_name = 'predictions'
    elif "multi_grey" in network_type:
    	stream="_1"
    	input_names = ['aa_input_1']
    	output_name = 'gv_smax'+stream
    #elif "mobilenet" in network_type:
    #	stream=""
    #	path2=""
    #	input_names = ['input_1']
    #	output_name = ['reshape_2']
    return input_names, output_name, stream

# Prepared Input For Model Training
def make_model_input_dict(input_names, batch):
    if len(input_names) == 1:
        fit_input = {input_names[0]: batch}
    else:
        fit_input = {input_names[0]:batch[0], input_names[1]:batch[1]}
    return fit_input

# Trains One Epoch Of The Network
def train_one_epoch(model, param_dict, input_names,output_name,
                    X_shuffled, Y_shuffled):
    batch_size = int(param_dict['batch_size'])
    l = len(X_shuffled)
    train_loss = 0
    train_acc = 0
    for i in range(0,l,batch_size):
        batch_end = min(i+batch_size,l)
        Xs = X_shuffled[i:batch_end]
        Ys = Y_shuffled[i:batch_end]
        batchsize = len(Xs)

        batch = gd.get_batch(Xs, param_dict)
        batch = da.augment_data(batch, param_dict, "train")
        
        #If single stream model, we have 1 input_name, otherwise 2
        fit_input = make_model_input_dict(input_names, batch)

        # Train on single batch
        history = model.fit(fit_input,{output_name: Ys},batch_size=batchsize)
        
        train_loss += float(history.history['loss'][0])*(batchsize/l)
        train_acc += float(history.history['acc'][0])*(batchsize/l)

    return model, train_loss, train_acc

# Performs Validation After An Epoch Has Completed
def validate_one_epoch(model, param_dict, input_names, output_name,
					   X_val, Y_val):
	X_val_data = gd.get_batch(X_val, param_dict)
	X_val_augmented = da.augment_data(X_val_data, param_dict, "val")

	# Evaluate on validation data
	print("Evaluating On Validation Data...")
	fit_input = make_model_input_dict(input_names, X_val_augmented)
	val_loss, val_acc = model.evaluate(fit_input,{output_name:Y_val})
	return val_loss, val_acc


# Checks If:
# - We Should Save Current Weights Due To Improved Validation Loss
# - We Should Reduce Learning Rate And Load In Previous Optimal Weights
def check_learning_rate(optimizer_type, model, savefile_name,output_name,
				   loss,best_loss,epoch,best_epoch, lr, counter,num_lr_changes):
    print("inside check learning")
    
    if loss > best_loss:
        counter+=1
        text_to_infofile="Loss not improved " + str(counter) + " time(s)\n"
        if (counter % 4) == 0:
            #Reduce learning rate and load best weights
            lr = lr*0.1
            num_lr_changes += 1
            optimizer = get_optimizer(optimizer_type, lr)
            model.compile(loss={output_name:'categorical_crossentropy'}, optimizer=optimizer, metrics=["accuracy"])
            print("Loading in weights from best run so far before reducing learning rate")
            text_to_infofile = "Reducing LR and loading Weights From Lowest Val Loss So Far" + '\n'
            model.load_weights("h5_weights/"+savefile_name+".h5", by_name=True)
    else:
        counter=0
        #Save weights because of decrease in val loss
        text_to_infofile="Saving Current Weights Due To Lower Validation Loss" + '\n'
        model.save_weights('h5_weights/' + savefile_name + '.h5')
        best_epoch = epoch
        best_loss = loss

    info_file = open("result_files/" + savefile_name + ".txt",'a')
    info_file.write(text_to_infofile)
    info_file.write("Current Learning Rate: " + str(lr) + '\n')
    info_file.write("Best Epoch: " + str(best_epoch) + ". Best Val Loss: " + str(best_loss) + '\n')

    return model, text_to_infofile, best_loss, best_epoch, lr,counter,num_lr_changes

# Constructs The Optimizer For The Model
def get_optimizer(optimizer_type, lr):
  if optimizer_type=='sgd':
    return optimizers.SGD(lr=lr, decay=1e-4, momentum=0.9, nesterov=False)
  elif optimizer_type=='adam':
    return optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

# Saves The Input Parameters Of The Network
def save_parameters(param_dict):
  info_file = open("result_files/" + param_dict['savefile_name'] + ".txt",'a')
  info_file.write('\n\n' + "NEW EXECUTION HERE" + '\n\n')
  for key in param_dict:
    print(key + ":", param_dict[key])
    info_file.write(key + ": " + param_dict[key] + '\n')
  info_file.close()

# Write Epoch Results To File
def write_epoch_results(filename, epoch, tloss, tacc, vloss, vacc):
  print("Epoch: ", epoch)
  print("Training Loss: ", str(tloss))
  print("Training Acc: ", str(tacc))
  print("Validation Loss: ", str(vloss))
  print("Validation Acc: " , str(vacc))

  info_file = open("result_files/" + filename + ".txt",'a')
  info_file.write("Epoch: " + str(epoch) + '\n')
  info_file.write("Training Loss: " + str(tloss) + '\n')
  info_file.write("Training Acc: " + str(tacc) + '\n')
  info_file.write("Val Loss: " + str(vloss) + '\n')
  info_file.write("Val Acc: " + str(vacc) + '\n')
