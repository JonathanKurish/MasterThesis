import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


# Points to what dataset function to use
def get_filenames(dataset_name, data_specifier, datatype, fewer=""):
	if data_specifier == "sample2":
		return get_sample_data2(datatype)

	elif dataset_name == "ucf101":
		return ucf101_get_filenames(int(data_specifier[4]), datatype)
	elif "b3sd" in dataset_name:
		if "fewer" in data_specifier:
			fewer="fewer_"
		return b3sd_get_filenames(int(data_specifier[0]), datatype, fewer)

# Returns the UCF101 filenames from the appropriate split
def ucf101_get_filenames(split=1, datatype="test"):
	X = []
	Y = []
	with open("ucf101_splits/new"+datatype+"list0" + str(split) + ".txt", 'r') as fp:
	    for line in fp:
	        [name, label] = line.split(" ")
	        X.append(name)
	        Y.append(int(label)-1)
	X = np.array(X)
	Y = np.array(Y)

	if datatype is not "test":
		Y = np.array(to_categorical(Y, num_classes=101))
	else:
		Y = Y[:,np.newaxis]
	return X, Y

# Load all filenames from B3SD dataset into list of names and classes.
def b3sd_get_filenames(classes=4, datatype="train", fewer=""):
	X = []
	Y = []
	start = "b3sd_splits/b3sd_"
	datafile = start + str(classes) + "_classes_" + fewer + datatype + ".txt"

	with open(datafile) as fp:
		for line in fp:
			[name, num] = line.split(" ")
			X.append(name)
			Y.append(int(num))

	num_inputs = len(X)
	if datatype!="test":
		Y = np.array(to_categorical(Y, num_classes=classes))
	else:
		Y = np.array(Y)[:,np.newaxis]
	return np.array(X), Y


#SKROT
def ucf_sample(split, datatype):
	X, Y = ucf101_get_filenames(split, datatype)
	X, Y = shuffle_data(X,Y)
	print(Y.shape)
	return X[0:20], Y[0:20,:]

#SKROT
from random import shuffle
def shuffle_data(X,Y):
  combined = list(zip(X, Y))
  shuffle(combined)
  X_shuffled, Y_shuffled = zip(*combined)
  return np.array(X_shuffled), np.array(Y_shuffled)



# SKAL SKROTTES TIL SIDST
def get_sample_data2(datatype, classes=101):
	X = []
	Y = []
	with open("ucf101_splits/sample_"+datatype+"2.txt", 'r') as fp:
	    for line in fp:
	        [name, label] = line.split(" ")
	        X.append(name)
	        Y.append(int(label)-1)

	Y_one_hot = to_categorical(Y, num_classes=classes)
	return np.array(X), np.array(Y_one_hot)
