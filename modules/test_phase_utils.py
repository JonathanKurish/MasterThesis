import numpy as np
import warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

  
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")



def training_file_parameters(train_file_name):
  param_dict = {}
  with open("result_files/"+train_file_name+".txt", 'r') as myfile:
      data = myfile.read()
      params = data.split("NEW EXECUTION HERE\n\n")[-1].split("\nE")[0]
      for prop_and_value in params.split("\n"):
        print(prop_and_value)
        prop, value = prop_and_value.split(": ")
        param_dict[prop] = value

  print(param_dict)
  return param_dict

def rename_param_dict(param_dict):
  new_param_dict={}
  new_param_dict['network_type'] = param_dict['Network Type']
  new_param_dict['channels'] = param_dict['Num Channels']
  new_param_dict['activation_function'] = param_dict['Activation Function']
  new_param_dict['dataset_name'] = param_dict['Dataset']
  new_param_dict['scale_weights_on_input'] = param_dict['Scale Weights Based On Frequency']
  new_param_dict['num_classes'] = param_dict['Num Output Classes']
  new_param_dict['sec1_frame'] = param_dict['Frame Chosen At 1 Second']
  new_param_dict['augment_data'] = param_dict['Augment Data']
  new_param_dict['depth'] = param_dict['Network Depth']
  new_param_dict['jumps'] = param_dict['With Jumps']
  new_param_dict['dropout'] = param_dict['Dropout Rate']
  new_param_dict['pooling'] = param_dict['Pooling Type Applied Late']
  new_param_dict['data_specifier'] = param_dict['Data Specifier']
  new_param_dict['initial_weights'] = param_dict['Initial Weights']
  new_param_dict['optimizer'] = param_dict['Optimizer']
  new_param_dict['initial_learning_rate'] = param_dict['Initial Learning Rate']
  new_param_dict['batch_size'] = param_dict['Batch Size']
  new_param_dict['reload_weights'] = param_dict['Reload Weights']
  new_param_dict['savefile_name'] = param_dict['Savefile Name']
  
  return new_param_dict

def top_n_predictions(x_predictions, n):
  # Get average prediction on the x frames and use as prediction
  x = np.mean(x_predictions, axis=0)
  
  print("Top " + str(n) + " percentages on the average prediction")
  top_n_classes = list(reversed([i for i in np.argsort(x)[-n:]]))
  top_n_pcts = list(reversed(x[np.argsort(x)[-n:]]))
  for i in range(0,n):
    print("#"+str(i+1)+" - Class "+str(top_n_classes[i]) + ", "+str(top_n_pcts[i])+"%")

  return top_n_classes

def calculate_accuracy(predictions, Y, num_inputs):
    correct_predictions = 0
    for idx, pred in enumerate(predictions):
        #Need to make Y[idx] = Y[idx]+1 if using UCF101
        if (Y[idx] == pred):
            correct_predictions += 1

    acc = float(correct_predictions) / float(num_inputs)
    print("Accuracy: " + str(acc) + " on " + str(num_inputs) + " inputs")
    return acc, correct_predictions

def plot_confusion_matrix(param_dict, Y, predictions, result_filename):
  conf_mat = confusion_matrix(Y, predictions)
  dataset = param_dict['dataset_name']
  data_specifier = param_dict['data_specifier']
  num_classes = int(param_dict['num_classes'])

  classes = []
  if dataset == "ucf101":
    with open("ucf101_splits/classInd.txt", 'r') as fp:
      for line in fp:
        classes.append(line[:-1])
    tick_marks = np.arange(4,len(classes),5)
    tick_marks_nums = np.arange(5,len(classes)+1,5)
    plt.xticks(tick_marks, tick_marks_nums, rotation=0, fontsize=8)
    plt.yticks(tick_marks, tick_marks_nums, fontsize=8)
    plt.grid(color='w', linestyle='-', linewidth=0.5)
  elif dataset == "b3sd":
    tick_marks = np.arange(0,num_classes)
    tick_marks_nums = np.arange(0,num_classes)
    plt.xticks(tick_marks, tick_marks_nums, rotation=0, fontsize=16)
    plt.yticks(tick_marks, tick_marks_nums, fontsize=16)
    plt.grid(color='w', linestyle='-', linewidth=0.5)



  theplot = plt.imshow(conf_mat, cmap='hot', interpolation='nearest')
  bar = plt.colorbar(theplot, orientation='vertical')
  plt.xlabel("Predicted Class")
  plt.ylabel("Actual Class")



  plt.savefig("confusion_matrices/"+result_filename+'.png',
              bbox_inches='tight',
              dpi=100)


  conf_mat = np.array(conf_mat)
  np.savetxt("confusion_matrices/" +result_filename+'conf_matrix.out', np.array(conf_mat).astype(int), fmt='%i', delimiter=',')


def save_results(filename, correct, test_acc, xlen, predlen):
    print("Predictions length:", predlen)
    print("X length:", xlen)
    print("Correct Guesses: ", correct)
    print("Test Accuracy: ", test_acc)

    result_file = open("test_results/" + filename + ".txt",'a')
    result_file.write("Predictions length: " + str(predlen) + "\n")
    result_file.write("X length: " + str(xlen) + "\n")
    result_file.write("Correct Guesses: " + str(correct) + "\n")
    result_file.write("Test Accuracy: " + str(test_acc) + "\n")
    result_file.close()

    return 0
