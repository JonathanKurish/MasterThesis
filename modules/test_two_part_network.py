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
filename_part1 = sys.argv[1]
filename_part2 = sys.argv[2]
repeats = 1

# Training parameters and test results are stored with the "_Test" extension
result_filename = "Two_Part_Network_Test"
result_file = open("test_results/" + result_filename + ".txt",'a')
param_dict_part1 = tpu.training_file_parameters(filename_part1)
param_dict_part2 = tpu.training_file_parameters(filename_part2)

# Some earlier training files were named differently, so we check for that
if 'network_type' not in param_dict_part1:
    param_dict_part1 = tpu.rename_param_dict(param_dict_part1)


# Change initial weights to be the result of training instead of
#    the initial weights in the training
param_dict_part1['initial_weights'] = filename_part1
param_dict_part2['initial_weights'] = filename_part2
param_dict_part1['scale_weights_on_input'] = "False"
param_dict_part2['scale_weights_on_input'] = "False"

# Write all parameters to terminal and result file
for key in param_dict_part1:
    value_part1 = param_dict_part1[key]
    value_part2 = param_dict_part2[key]

    print(key + ":", value_part1, "\t", value_part2)
    result_file.write(key+": " + value_part1+"\t"+value_part2+'\n')

#param_dict_part1['dataset_name'] = "d:/b3sd"
#param_dict_part2['dataset_name'] = "d:/b3sd"

# Get parameters from training parameters
network_type2 = param_dict_part1['network_type']
dataset_name2 = param_dict_part1['dataset_name']
print(dataset_name2)
data_specifier2 = param_dict_part1['data_specifier']
sec1_frame2 = tpu.str2bool(param_dict_part1['sec1_frame'])

# Get parameters from training parameters
network_type4 = param_dict_part2['network_type']
dataset_name4 = param_dict_part2['dataset_name']
data_specifier4 = param_dict_part2['data_specifier']
sec1_frame4 = tpu.str2bool(param_dict_part2['sec1_frame'])


# Read filenames for dataset to use
X2,Y2 = gf.get_filenames("b3sd", "2_classes", "test")
#dataset = gf.b3sd_sample(int(data_specifier[0]), "test", "")
#dataset = gf.ucf_sample(int(data_specifier[4]), "test")
#dataset = gf.b3sd_local_sample(param_dict, "test")
print(X2.shape)
print(Y2.shape)

# Prepare input and output name of model,
#   then make model and load weights from training
# Weights are saved before and efter to validate succesful load
input_part1, op1, stream_part1 = tu.prepare_stream(network_type2)    
model_part1 = tu.make_network_model(param_dict_part1,stream_part1,None)


weights_pre = model_part1.get_weights()
model_part1 = tu.load_weights(model_part1, param_dict_part1, stream_part1, True, "test")
weights_after = model_part1.get_weights()
weight_names = [weight.name for layer in model_part1.layers for weight in layer.weights]
tu.check_weights(weights_pre, weights_after, weight_names)


# If the frame at second 1 is used (time of shot in B3SD dataset), then
#  we only get that particular frame for each video.

print("1sec_frame set, so choosing that frame for each video")
result_file.write("1sec_frame set, so choosing that frame for each video\n")
X2_data = gd.get_batch(X2, param_dict_part1)
augmented_X2 = da.augment_data(X2_data, param_dict_part1, "test")


fit_input2 = tu.make_model_input_dict(input_part1, augmented_X2)
predictions_categorical2 = model_part1.predict(fit_input2, verbose=1)
    
print(predictions_categorical2.shape)
predictions2 = np.asarray([np.argmax(pred) for pred in predictions_categorical2])
print(predictions2.shape)
acc2, correct_predictions2 = tpu.calculate_accuracy(predictions2, Y2, len(predictions2))


correct=0
ones = 0
actual_shots=0
actual_no_shots=0
shot_guesses=[]
new_Y=[]
correct_noshots1=0
correct_shots=0
wrong_no_shots=0
wrong_shots=0
for i,pred in enumerate(predictions2):
    pred = int(pred)
    y = int(Y2[i])
    if pred==0:
        if y==0:
            correct_noshots1+=1
            correct+=1
        else:
            wrong_no_shots+=1

    if pred==1:
        shot_guesses.append(X2[i])
        new_Y.append(Y2[i])
        if y==1:
            ones+=1
            correct_shots+=1
            correct+=1
        else:
            wrong_shots+=1
print("Shot / No Shot results:")
print("Total Num Clips:", len(predictions2))
print("Number of Correct Shot Classifications:", correct_shots)
print("Number of Correct No Shot Classifications:", correct_noshots1)
print("Number of Shots That Ended As a No Shot Classification:", wrong_no_shots)
print("Number of No Shots That Ended As a Shot Classification:", wrong_shots)
print("Accuracy on the inputs so far:", correct_noshots1/len(predictions2))


new_Y2 = []
for i,x in enumerate(shot_guesses):
    if "layup" in x:
        new_Y2.append(0)
    elif "freethrow" in x:
        new_Y2.append(1)
    elif "point-2" in x:
        new_Y2.append(2)
    elif "point-3" in x:
        new_Y2.append(3)
    elif "no-shot" in x:
        new_Y2.append(4)


input_names4, output_name4, stream4 = tu.prepare_stream(network_type4)    
model4 = tu.make_network_model(param_dict_part2,stream4,None)


weights_pre = model4.get_weights()
model4 = tu.load_weights(model4, param_dict_part2, stream4, True, "test")
weights_after = model4.get_weights()
weight_names = [weight.name for layer in model4.layers for weight in layer.weights]
tu.check_weights(weights_pre, weights_after, weight_names)

X4 = gd.get_batch(shot_guesses, param_dict_part2)
augmented_X4 = da.augment_data(X4, param_dict_part2, "test")


fit_input4 = tu.make_model_input_dict(input_names4, augmented_X4)
predictions_categorical4 = model4.predict(fit_input4, verbose=1)

#print(predictions_categorical4)
predictions4 = np.asarray([np.argmax(pred) for pred in predictions_categorical4])
print(predictions4.shape)

acc4, correct_predictions4 = tpu.calculate_accuracy(predictions4, new_Y2, len(predictions4))




correct=0
ones = 0
actual_shots=0
actual_no_shots=0
noshots=0
correct_noshots=0
correct_shots=0
for i,pred in enumerate(predictions4):
    pred = int(pred)
    y = int(new_Y2[i])
    if pred==y:
        correct+=1
    #print("guess:", pred, "truth:", y)
    '''elif pred==0 and y==0:
                    noshots+=1
                    correct_noshots+=1
                    correct+=1
                if pred==1 and y==1:
                    ones+=1
                    correct_shots+=1
                    correct+=1
                if y==1:
                    actual_shots+=1
                else:
                    actual_no_shots+=1'''

print("Number of Correct Classifications In Part 2:", correct)
'''print("Number of Correct Shot Classifications:", correct_shots)
print("Number of Correct No Shot Classifications:", correct_noshots)
print("Number of Actual Shots", actual_shots)
print("Number of Actual No Shots:", actual_no_shots)
print("Number of Guessed Shots:", ones)
print("Number of Guessed No Shots:", noshots)'''
print("Second Part Accuracy:", correct/len(predictions4))
print("Final Accuracy:", (correct+correct_noshots1)/len(predictions2))


#print("correct:", correct, "incorrect:", len(Y2), "ones:", ones, "ao:",actual_ones)

# Plot confusion matrix
tpu.plot_confusion_matrix(param_dict_part2, new_Y-1, np.asarray(predictions4)-1, result_filename)




# Save test results to file
#end = tpu.save_results(result_filename, correct_predictions2, acc2, len(X2), len(predictions2))