import time
import sys
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


# Own Imports
import get_filenames as gf
import train_utils as tu



# Reads variables specified in the terminal and places them
# in a dictionary, param_dict
param_dict = {}
for i in sys.argv[1:]:
  variable, value = i.split("=")
  param_dict[variable] = value

# Write parameters to output file given amongst parameters
tu.save_parameters(param_dict)

# Read filenames for dataset to use
dataset_name = param_dict['dataset_name']
data_specifier = param_dict['data_specifier']
X,Y = gf.get_filenames(dataset_name,data_specifier,"train")
X_val,Y_val = gf.get_filenames(dataset_name,data_specifier,"val")

# Get distribution of classes, potentially to be used in training
class_distribution = tu.get_class_distribution(Y)

# Read some parameters needed for training
savefile_name = param_dict['savefile_name']
optimizer_type = param_dict['optimizer']
current_lr = float(param_dict['initial_learning_rate'])



    
# Prepare input and output name of model,
# then make model and load initial weights
input_names, output_name, stream = tu.prepare_stream(param_dict['network_type'])
model = tu.make_network_model(param_dict, stream, class_distribution)
weights_pre = model.get_weights()
model = tu.load_weights(model, param_dict, stream, True, "train")
weights_after = model.get_weights()
weight_names = [weight.name for layer in model.layers for weight in layer.weights]
tu.check_weights(weights_pre, weights_after, weight_names)

# Set optimizer
optimizer = tu.get_optimizer(optimizer_type, current_lr)
model.compile(loss={output_name:'categorical_crossentropy'},
              optimizer=optimizer,
              metrics=["accuracy"])

print("Total number of parameters in network:",model.count_params())

# Variable initialization needed for training
previous_validation_loss = best_validation_loss = 9999
epoch = counter = num_lr_changes = best_epoch = 0
keep_training = True
X_shuffled, Y_shuffled = X, Y

# Network training
while keep_training:
    # Increment epoch variable
    epoch += 1

    # Time each epoch
    start_time = time.time()

    # Print current epoch and learning rate
    print("Epoch: " + str(epoch) + ", Current Learning Rate: ", current_lr)

    # Re-shuffling of data in each epoch
    X_shuffled, Y_shuffled = tu.shuffle_data(X_shuffled,Y_shuffled)


    model, train_loss, train_acc = tu.train_one_epoch(model, param_dict, input_names,output_name,
                    X_shuffled, Y_shuffled)

    val_loss, val_acc = tu.validate_one_epoch(model, param_dict,
        input_names, output_name, X_val, Y_val)

    # Write epoch results to result file
    tu.write_epoch_results(savefile_name,epoch,train_loss, train_acc, val_loss, val_acc)
    

    # Save weights if validation loss is a new best.
    # Reload previous best and reduce learning rate
    #   if time x consecutive unimproved epochs.
    tmp = tu.check_learning_rate(optimizer_type, model,savefile_name, output_name,val_loss,
      best_validation_loss, epoch, best_epoch, current_lr, counter, num_lr_changes)
    model, text_to_infofile, best_validation_loss, best_epoch, current_lr,counter, num_lr_changes = tmp
       
    # Stop training if 4th time reducing learning rate
    if (num_lr_changes>2):
        keep_training = False

    
    # Save epoch time to result file
    epoch_time = (time.time() - start_time)/60
    print("epoch_time:", str(epoch_time))
    info_file = open("result_files/" + param_dict['savefile_name'] + ".txt",'a')
    info_file.write("Epoch Time In Minutes: " + str(epoch_time) + '\n')
    info_file.close()

# Write "Training Completed" to result file
info_file = open("result_files/" + savefile_name + ".txt",'a')
info_file.write("\nTraining Completed\n")
info_file.close()