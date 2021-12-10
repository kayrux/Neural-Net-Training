import numpy as np
import network
import time

# converts a 1d python list into a (1,n) row vector
def rv(vec):
    return np.array([vec])
    
# converts a 1d python list into a (n,1) column vector
def cv(vec):
    return rv(vec).T
        
# creates a (size,1) array of zeros, whose ith entry is equal to 1    
def onehot(i, size):
    vec = np.zeros(size)
    vec[i] = 1
    return cv(vec)

# returns a list containing the pixels for each image, stored as a (784, 1) numpy array
def prepArray(arrayImages):

    # We want to flatten each image from a 28 x 28 to a 784 x 1 numpy array
    # CODE GOES HERE
    features = []
    for row in arrayImages:
        featureVec = cv(row.flatten())
        featureVec = featureVec / 255
        features.append(featureVec)
    
    print(np.shape(features))
    # convert to floats in [0,1] (only really necessary if you have other features, but we'll do it anyways)
    # CODE GOES HERE
    return features
  
#################################################################

# reads the data from the notMNIST.npz file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    # loads the four arrays specified.
    # train_features and test_features are arrays of (28x28) pixel values from 0 to 255.0
    # train_labels and test_labels are integers from 0 to 9 inclusive, representing the letters A-J
    with np.load("data/notMNIST.npz", allow_pickle=True) as f:
        train_features, train_labels = f['x_train'], f['y_train']
        test_features, test_labels = f['x_test'], f['y_test']
        
    # need to rescale, flatten, convert training labels to one-hot, and zip appropriate components together
    # CODE GOES HERE
    training_features = prepArray(train_features)
    testing_features = prepArray(test_features)
    trainingLabels = [onehot(label, 10) for label in train_labels]


    trainingData = zip(training_features, trainingLabels)
    testingData = zip(testing_features, test_labels)
    return (trainingData, testingData)
       
    
###################################################################


trainingData, testingData = prepData()

startTime = time.time_ns()
net = network.loadFromFile("part2.pkl")
# net = network.Network([784, 30, 20, 10])
net.SGD(trainingData, 20, 10, 1.5, test_data = testingData)
print(f"Time elapsed: {(time.time_ns() - startTime) / 1000000000} seconds")
# network.saveToFile(net, "part2.pkl")







