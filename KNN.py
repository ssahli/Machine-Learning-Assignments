import numpy as np
from scipy import stats
from sklearn import metrics
from matplotlib import pyplot as plt
import matplotlib.cm as cm

k = 30

'''
    Data
        > n = number of features
        > m = number of training examples
'''
train = np.loadtxt('optdigits/optdigits.train', delimiter=',')
test = np.loadtxt('optdigits/optdigits.test', delimiter=',')
train_target = train[:,-1]
test_target = test[:,-1]
train = train[:,:-1]
test = test[:,:-1]
n = train.shape[1]
m = train.shape[0]





'''
    x: data set
    y: centroids
'''
def euclidean(x, y):
    return np.sqrt(np.sum((x - y) ** 2, axis=1))

'''
    x: data set
    y: centroids
    assign: cluster assignments
'''
def SSE(x, y, assign):
    sse = 0
    for i in range(k):
        sse += np.sum(euclidean(x[assign == i,:], y[i]) ** 2)
    return sse

'''
    c: centroids
'''
def SSS(c):
    sss = 0
    for i in range(k):
        sss += np.sum(euclidean(c, c[i]) ** 2)
    return sss

'''
    x: data set
    c: centroids
    label: labels of the data
    assign: cluster assignments
'''
def ME(x, c, label, assign):
    entropies = np.zeros(k)
    proportions = np.zeros(10)
    mean_entropy = 0
    # Mean entropy
    for i in range(k):
        # Entropy
        for j in range(10):
            proportions[j] = float((x[np.logical_and((assign == i), (label == j))].shape[0])) / (x[assign == i].shape[0])
        proportions[proportions == 0.] = 0.000000000000000001
        entropies[i] = -np.sum(proportions * np.log2(proportions))
        mean_entropy += entropies[i] * x[assign == i].shape[0]
    mean_entropy /= m
    return mean_entropy





'''
    k-means
        > Choose 10 random centroids
        > Perform k-means 5 times:
            > Calculate euclidean distances between centroids and data points
            > Assign a centroid to every data point
            > Repeat until centroids stop moving
            > Calculate SSE
            > Save the centroids with the lowest SSE out of the 5 runs
'''
sse = 0
best_sse = 0
centroids = np.zeros((m,k))
best_centroids = np.zeros(centroids.shape)
best_assigned = np.zeros(m)

for j in range(5):
    # Choose 10 random centroids
    centroids = train[np.random.choice(m, k)]
    distances = np.zeros((m, k))
    assigned = np.zeros(m)

    # Perform k-means, updating the distances, centroids, and assignments
    for _ in range(100):
        for i in range(k):
            distances[:,i] = euclidean(train, centroids[i])
        assigned = np.argmin(distances, axis=1)
        for i in range(k):
            centroids[i] = np.mean(train[assigned == i,:], axis=0)
    for i in range(k):
        distances[:,i] = euclidean(train, centroids[i])
    assigned = np.argmin(distances, axis=1)

    # Calculate SSE
    sse = SSE(train, centroids, assigned)
    print "SSE of trial " + str(j+1) + ": " + str(sse)

    # Save the best trial
    if sse < best_sse or best_sse == 0:
        best_sse = sse
        best_centroids = centroids.copy()
        best_assigned = assigned.copy()

print "SSE of the best model: " + str(best_sse)
print "SSS of the best model: " + str(SSS(best_centroids))
print "Mean Entropy of best model: " + str(ME(train, best_centroids, train_target, best_assigned))





'''
    > Calculate train accuracy
    > Calculate test accuracy
    > Create a confusion matrix
    > Visualize the clusters
'''
# Calculate train accuracy
train_predictions = np.zeros(m)
centroid_predictions = np.zeros(k)
for i in range(k):
    centroid_predictions[i] = stats.mode(train_target[assigned == i])[0]
    train_predictions[assigned == i] = centroid_predictions[i]
accuracy = metrics.accuracy_score(train_target, train_predictions)
print "Accuracy on train set: " + str(accuracy)



# Calculate test accuracy
distances = np.zeros((test.shape[0], k))
test_predictions = np.zeros(test.shape[0])
centroid_predictions = np.zeros(k)
test_assigned = np.zeros(test.shape[0])
for i in range(k):
    centroid_predictions[i] = stats.mode(train_target[assigned == i])[0]
    distances[:,i] = euclidean(test, centroids[i])
test_assigned = np.argmin(distances, axis=1)
for i in range(k):
    test_predictions[test_assigned == i] = centroid_predictions[i]
accuracy = metrics.accuracy_score(test_target, test_predictions)
print "Accuracy on test set: " + str(accuracy)



# Create a confusion matrix
confusion = metrics.confusion_matrix(test_target, test_predictions)
print confusion



# Visualize the clusters
for i in range(k):
        plt.imsave('image%i.png' %i, np.array(centroids[i]).reshape(8,8), cmap=cm.gray)
