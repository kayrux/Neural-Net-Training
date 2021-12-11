import csv
import numpy as np
from numpy.core.fromnumeric import std
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

# Calculates the mean, standard deviation, and max age from the given array
# Returns a tuple of (means, std_devs, max_age)
def get_means_stdDevs_maxAge(features):
    means = []
    std_devs = []
    sbp = []
    tobacco = []
    ldl = []
    adiposity = []
    typea = []
    obesity = []
    alcohol = []
    max_age = 0

    for row in features:
        sbp.append(float(row[0]))
        tobacco.append(float(row[1]))
        ldl.append(float(row[2]))
        adiposity.append(float(row[3]))
        typea.append(float(row[5]))
        obesity.append(float(row[6]))
        alcohol.append(float(row[7]))
        if(float(row[8]) > max_age): max_age = float(row[8])

    means.append(np.mean(sbp))
    std_devs.append(np.std(sbp))

    means.append(np.mean(tobacco))
    std_devs.append(np.std(tobacco))

    means.append(np.mean(ldl))
    std_devs.append(np.std(ldl))

    means.append(np.mean(adiposity))
    std_devs.append(np.std(adiposity))

    means.append(0.0)
    std_devs.append(0.0)

    means.append(np.mean(typea))
    std_devs.append(np.std(typea))

    means.append(np.mean(obesity))
    std_devs.append(np.std(obesity))

    means.append(np.mean(alcohol))
    std_devs.append(np.std(alcohol))
    
    return means, std_devs, max_age

# reads the data from the heart.csv file,
# divides the data into training and testing sets, and encodes the training vectors in onehot form
# returns a tuple (trainingData, testingData), each of which is a zipped array of features and labels
def prepData():
# - Family history should be stored as a boolean
# - Age column should be rescaled
# - All other rows should be converted to z scores
# - You’ll have to calculate the average and standard deviation yourself from the
#   data in the file
    feature_vectors = []
    n, features, labels = readData('data/heart.csv')
    means, std_devs, max_age = get_means_stdDevs_maxAge(features)

    for row in features:
        # print(row)
        sbp = standardize( float(row[0]), means[0], std_devs[0])
        tobacco = standardize(float(row[1]), means[1], std_devs[1])
        ldl = standardize(float(row[2]), means[2], std_devs[2])
        adiposity = standardize(float(row[3]), means[3], std_devs[3])
        famhist = 0.0
        if (row[4] == "Present"): famhist = 1.0
        typea = standardize(float(row[5]), means[5], std_devs[5])
        obesity= standardize(float(row[6]), means[6], std_devs[6])
        alcohol = standardize(float(row[7]), means[7], std_devs[7])
        age = float(row[8]) / max_age
        feature_vectors.append(cv([sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age]))
        #print(cv([sbp, tobacco, ldl, adiposity, famhist, typea, obesity, alcohol, age]))

    print(np.shape(feature_vectors))
    

    n_train = int (n * 5/6)
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


       