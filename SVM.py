import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import preprocessing
from sklearn.svm import SVC

'''
    Load and process the data. Note the last column is the response
        > Separate positive and negative examples
        > Cut the two new sets in half
        > Concatenate to make two sets of equal pos/neg examples
            (Note that this causes a lot of examples to be dropped if the data
             has more of one class over the other)
        > Standardize the data based off mean and std deviation of the train set
        > Shuffle the training set
'''
data = np.loadtxt('spambase.txt', delimiter=',')
print 'Processing data...'
pos = data[data[:,-1] == 1]
neg = data[data[:,-1] == 0]
if len(pos) > len(neg):
    pos = pos[:len(neg)]
else:
    neg = neg[:len(pos)]
pos1 = pos[:len(pos)/2]
pos2 = pos[len(pos)/2:]
neg1 = neg[:len(neg)/2]
neg2 = neg[len(neg)/2:]
train_set = np.vstack((pos1, neg1))
test_set = np.vstack((pos2, neg2))
scaler = preprocessing.StandardScaler().fit(train_set[:,:-1])
train_set[:,:-1] = scaler.transform(train_set[:,:-1])
test_set[:,:-1] = scaler.transform(test_set[:,:-1])
np.random.shuffle(train_set)





'''
    Experiment 1: k-folds cross-validation to determine the best C value
    k = number of folds for k-folds cross validation
'''
k = 10
splits = np.array_split(train_set, k)

print "\nExperiment 1: Cross-validation"
print "Performing k-folds to obtain best C-value..."
accuracy = 0
for j in range(10):
    '''
        Initialize an SVM where C = j/10 (.1, .2,..., 1.)
        ScikitLearn doesn't allow for C = 0.0, so it is skipped
            1) Run k-folds for a given value of C
            2) If the average accuracy is greater than the previous,
               save both the accuracy and the corresponding C value
    '''
    C = float(j+1)/10
    SVM = SVC(kernel='linear', C=C)
    previous_accuracy = accuracy
    accuracy = 0

    for i in range(k):
        '''
            The following three lines:
                1) Set the current train set equal to the 10 training sets
                   (this way we don't alter the 10 sets themselves)
                2) Remove the one set that's used for validation and store
                   it in current_test
                3) Re-combine the current_train list of matrices into a
                   single matrix (so they can be fed into SVM.fit())
        '''
        current_train = list(splits)
        current_test = current_train.pop(i)
        current_train = np.vstack(current_train)
        SVM.fit(current_train[:,:-1], current_train[:,-1])
        predicted = SVM.predict(current_test[:,:-1])
        accuracy += metrics.accuracy_score(current_test[:,-1], predicted)

    accuracy /= k
    if accuracy > previous_accuracy:
        best_accuracy = accuracy
        best_C = C

print "Best training accuracy: " + str(best_accuracy)
print "Best C-value: " + str(best_C)

'''
    Train a new SVM on the full training set and test it on the test set.
    This SVM returns a probability rather than a binary classification. This
    way, we can manually set our own threshold for classification. The default
    threshold for the SVM used in k-folds is 0.5, and returns a prediction
    accordingly.

    Keep both the binary prediction and the probabilities, since the ROC
    curve requires multiple different thresholds to calculate.
'''
print "Training on full set..."
target = test_set[:,-1]
SVM = SVC(kernel='linear', C=best_C, probability=True)
SVM.fit(train_set[:,:-1], train_set[:,-1])
probabilities = SVM.predict_proba(test_set[:,:-1])
probabilities = probabilities[:,1]
predicted = np.copy(probabilities)
predicted[predicted >= 0.5] = 1.
predicted[predicted < 0.5] = 0.
accuracy = metrics.accuracy_score(target, predicted)
precision = metrics.precision_score(target, predicted)
recall = metrics.recall_score(target, predicted)

print "Accuracy on test set: " + str(accuracy)
print "Precision: " + str(precision)
print "Recall: " + str(recall)

'''
    Plot the ROC curve
    Recalculate the predictions based on the current threshold. Then, add up
    the false positives and true negatives to calculate false positive rate.
    Grab the recall (= true positive rate) and plot the ROC curve.
'''
thresholds = 200
threshold = 1./thresholds
tpr_array = np.zeros(thresholds)
fpr_array = np.zeros(thresholds)
baseline = np.zeros(thresholds)

print "Generating ROC curve..."
for i in range(thresholds):
    current_threshold = (i+1) * threshold
    predicted = np.copy(probabilities)
    predicted[predicted >= current_threshold] = 1.
    predicted[predicted < current_threshold] = 0.
    fp = np.sum(predicted > target)
    tn = np.sum((predicted == 0) & (target == 0))
    fpr = fp / (fp + tn * 1.)
    tpr = metrics.recall_score(target, predicted)
    tpr_array[i] = tpr
    fpr_array[i] = fpr
    baseline[i] = current_threshold

plt.plot(baseline, baseline, 'r--')
plt.plot(fpr_array[::-1], tpr_array[::-1])
plt.title("ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.axis([0,1,0,1])
plt.show()





'''
    Experiment 2: Feature selection
    The weights from the SVM is a column vector, so the transpose is taken and
    is sorted from largest to smallest.
'''
weights = SVM.coef_
weights = np.squeeze(np.asarray(weights))
sort = np.sort(weights)
sort = sort[::-1]
m = weights.shape[0]
m_array = np.zeros(m)
accuracies = np.zeros(m)

print "\nExperiment 2: Feature selection"
print "Calculating accuracies on sets of features..."
for i in range(2,m):
    selected = sort[:i]
    selected = np.nonzero(np.in1d(weights,selected))
    selected = np.squeeze(np.asarray(selected))
    SVM = SVC(kernel='linear', C=best_C)
    SVM.fit(train_set[:,selected], train_set[:,-1])
    predicted = SVM.predict(test_set[:,selected])
    accuracies[i] = metrics.accuracy_score(target, predicted)
    m_array[i] = i
    if i == 5:
        best_features = selected

print "Indeces of top 5 features: "
print best_features

print "Generating accuracy plot..."
plt.plot(m_array, accuracies)
plt.title("Accuracies for m selected features")
plt.xlabel("m")
plt.ylabel("accuracy")
plt.axis([0,m,0,1])
plt.show()





'''
    Experiment 3
'''

print "\nExperiment 3: Random feature selection"
print "Calculating accuracies on sets of features..."
for i in range(2,m):
    selected = np.random.randint(0,m,i)
    SVM = SVC(kernel='linear', C=best_C)
    SVM.fit(train_set[:,selected], train_set[:,-1])
    predicted = SVM.predict(test_set[:,selected])
    accuracies[i] = metrics.accuracy_score(target, predicted)

print "Generating accuracy plot..."
plt.plot(m_array, accuracies)
plt.title("Accuracies for m random features")
plt.xlabel("m")
plt.ylabel("accuracy")
plt.axis([0,m,0,1])
plt.show()
