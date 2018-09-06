import numpy as np
from keras.models import Model
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')
import data_augmentation as da
import test_phase_utils as tpu
import train_utils as tu
import sys
import get_filenames as gf
import get_data as gd

# Read the file we are going to do test on
train_file_name = sys.argv[1]
repeats = 5

# Training parameters and test results are stored with the "_Test" extension
result_filename = train_file_name + "_Test"
result_file = open("test_results/" + result_filename + ".txt",'a')
param_dict = tpu.training_file_parameters(train_file_name)

# Some earlier training files were named differently, so we check for that
if 'network_type' not in param_dict:
  param_dict = tpu.rename_param_dict(param_dict)

# Change initial weights to be the result of training instead of
#    the initial weights in the training
param_dict['initial_weights'] = train_file_name
param_dict['scale_weights_on_input'] = "False"

# Write all parameters to terminal and result file
for key in param_dict:
  print(key + ":", param_dict[key])
  result_file.write(key + ": " + param_dict[key] + '\n')

#param_dict['dataset_name'] = "d:/b3sd"


# Get parameters from training parameters
network_type = param_dict['network_type']
dataset_name = param_dict['dataset_name']
data_specifier = param_dict['data_specifier']
#channels = [int(c) for c in param_dict['channels'].split(",")]
sec1_frame = tpu.str2bool(param_dict['sec1_frame'])


# Read filenames for dataset to use
X,Y = gf.get_filenames(dataset_name, data_specifier, "test")
print(X.shape)
print(Y.shape)

# Prepare input and output name of model,
#   then make model and load weights from training
# Weights are saved before and efter to validate succesful load
input_names, output_name, stream = tu.prepare_stream(network_type)    
model = tu.make_network_model(param_dict,stream,None)

weights_pre = model.get_weights()
model = tu.load_weights(model, param_dict, stream, False, "test")
weights_after = model.get_weights()
weight_names = [weight.name for layer in model.layers for weight in layer.weights]
tu.check_weights(weights_pre, weights_after, weight_names)


# If the frame at second 1 is used (time of shot in B3SD dataset), then
#  we only get that particular frame for each video.

if sec1_frame:
    print("1sec_frame set, so choosing that frame for each video")
    result_file.write("1sec_frame set, so choosing that frame for each video\n")
    X = gd.get_batch(X, param_dict)
    augmented_X = da.augment_data(X, param_dict, "test")


    fit_input = tu.make_model_input_dict(input_names, augmented_X)
    predictions_categorical = model.predict(fit_input, verbose=1)
    
    print(predictions_categorical.shape)
    predictions = np.asarray([np.argmax(pred) for pred in predictions_categorical])
    print(predictions.shape)

    acc, correct_predictions = tpu.calculate_accuracy(predictions, Y, len(predictions))

# If sec1_frame is false, then we select x frames from each video for a 
#   more averaged guess for each video.
else:
    print("Augmenting each input "+str(repeats)+" times")
    result_file.write("Augmenting each input "+str(repeats)+" times\n")
    result_file.close()
    
    # Prepare variables for testing
    predictions=[]
    for idx, x in enumerate(X):
        if (idx%10) == 0:
            print("Now working on test input number", idx)
        
        # Get x frames from the same video
        #x_frames = gd.get_X_frames(x, param_dict, 10)
        x_frames = gd.get_batch(x, param_dict, phase="test", repeats=repeats)
        x_frames = da.augment_data(x_frames, param_dict, "test")

        # Predict on each of the frames
        fit_input = tu.make_model_input_dict(input_names, x_frames)
        x_predictions_on_same_input = model.predict(fit_input, verbose=1)

        print("\nInput", idx, "- Truth:", x, int(Y[idx][0]))
        # Returns the top n predicted classes over an average prediction
        top_preds = tpu.top_n_predictions(x_predictions_on_same_input, 2)

        # Add prediction to list of predictions
        predictions.append(top_preds[0])

        # Calculate test accuracy
        acc, correct_predictions = tpu.calculate_accuracy(predictions, Y, idx+1)


# Plot confusion matrix
tpu.plot_confusion_matrix(param_dict, Y, np.asarray(predictions)-1, result_filename)

# Save test results to file
end = tpu.save_results(result_filename, correct_predictions, acc, len(X), len(predictions))