import csv
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

# given a data point, mean, and standard deviation, returns the z-score
def standardize(x, mu, sigma):
    return ((x - mu)/sigma)
    

##############################################

# reads number of data points, feature vectors and their labels from the given file
# and returns them as a tuple
def readData(filename):

    # CODE GOES HERE
    n = 0
    features = []
    labels = []
    with open("data/heart.csv", "rt") as f:
        
        data = csv.reader(f)
        for row in data:
            if (n == 0): 
                n += 1
                continue
            features.append(cv(row[1:10]))
            #print(type(row[10]))
            labels.append(float(row[10]))
            n += 1
    

    # print(labels)
    return n - 1, features, labels


################################################

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
    sbp_mean = 138.3
    sbp_std_dev = 20.5
    tobacco_mean = 3.64
    tobacco_std_dev = 4.59
    ldl_mean = 4.74
    ldl_std_dev = 2.07
    adiposity_mean = 25.4
    adiposity_std_dev = 7.77
    typea_mean = 53.1
    typea_std_dev = 9.81
    obesity_mean = 26.0
    obesity_std_dev = 4.21
    alcohol_mean = 17.0
    alcohol_std_dev = 24.5
    max_age = 64
# - Family history should be stored as a boolean
# - Age column should be rescaled
# - All other rows should be converted to z scores
# - You’ll have to calculate the average and standard deviation yourself from the
#   data in the file
    feature_vectors = []
    n, features, labels = readData('data/heart.csv')
    for row in features:
        # print(row)
        sbp = standardize( float(row[0]), sbp_mean, sbp_std_dev)
        tobacco = standardize(float(row[1]), tobacco_mean, tobacco_std_dev)
        ldl = standardize(float(row[2]), ldl_mean, ldl_std_dev)
        adiposity = standardize(float(row[3]), adiposity_mean, adiposity_std_dev)
        famhist = 0.0
        if (row[4] == "Present"): famhist = 1.0
        typea = standardize(float(row[5]), typea_mean, typea_std_dev)
        obesity= standardize(float(row[6]), obesity_mean, obesity_std_dev)
        alcohol = standardize(float(row[7]), alcohol_mean, alcohol_std_dev)
        age = float(row[8]) / max_age
        feature_vectors.append(cv([sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age]))
        #print(cv([sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age]))

    print(np.shape(feature_vectors))
    

    n_train = int (n * 5/6)
    print(n_train)
    trainingFeatures = feature_vectors[:n_train]
    trainingLabels = [onehot(int(label), 2) for label in labels[:n_train]]
    

    testingFeatures = feature_vectors[n_train:]
    testingLabels = labels[n_train:]
    # print(f"Number of training samples: {n_train}")

    trainingData = zip(trainingFeatures, trainingLabels)
    testingData = zip(testingFeatures, testingLabels)
    # print(f"Number of testing samples: {n - n_train}")
    
    # CODE GOES HERE

    return (trainingData, testingData)


###################################################


trainingData, testingData = prepData()

net = network.Network([9,10,2])
startTime = time.time_ns()
net.SGD(trainingData, 10, 10, .1, test_data = testingData)
print(f"Time elapsed: {(time.time_ns() - startTime) / 1000000000} seconds")


       