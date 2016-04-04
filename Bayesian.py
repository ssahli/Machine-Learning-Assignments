import numpy as np
from sklearn import metrics

'''
    1. Create training and test set
        > Each set has approx 1/2 of examples
        > Each set has proportional number of positive/negative examples
'''
data = np.loadtxt('spambase.txt', delimiter=',')
np.random.shuffle(data)
pos = data[data[:,-1] == 1]
neg = data[data[:,-1] == 0]
pos1 = pos[:len(pos)/2]
pos2 = pos[len(pos)/2:]
neg1 = neg[:len(neg)/2]
neg2 = neg[len(neg)/2:]
train_set = np.vstack((pos1, neg1))
test_set = np.vstack((pos2, neg2))





'''
    2. Create probabalistic model
        > Calculate probability of spam/not spam via mean
        > Calculate mean of every column based off class
        > Calculate standard deviations of every column based off class
        > If by chance there is a standard deviation equal to 0, set it
          to a really low value (eg 0.000001)
'''
pSpam = np.mean(train_set[:,-1])
pNotSpam = 1 - pSpam

mu_pos = np.mean(pos1[:,:-1], axis=0)
mu_neg = np.mean(neg1[:,:-1], axis=0)

sigma_pos = np.std(pos1[:,:-1], axis=0)
sigma_neg = np.std(neg1[:,:-1], axis=0)

sigma_pos[sigma_pos == 0.] = .000001
sigma_neg[sigma_neg == 0.] = .000001





'''
    3. Run Naive Bayes on the test data
        > Calculate probabilities for positive and negative hypotheses
        > Take the log of the probabilities and product them
        > Construct a list of predictions
        > Calculate accuracy, precision, recall
        > Construct confusion matrix
'''

def gaussian(mu, sigma, X):
    y = np.zeros(X.shape)
    exp = np.exp(-0.5 * (((X - mu) ** 2.) / (sigma ** 2)))
    y = (1. / ((np.sqrt(2. * np.pi)) * sigma)) * exp
    return y

pPositive = gaussian(mu_pos, sigma_pos, test_set[:,:-1])
pNegative = gaussian(mu_neg, sigma_neg, test_set[:,:-1])

pPositive = np.log(pSpam * np.prod(pPositive, axis=1))
pNegative = np.log(pNotSpam * np.prod(pNegative, axis=1))

predictions = np.zeros(pPositive.shape)
for i in range(len(predictions)):
    if pPositive[i] > pNegative[i]:
        predictions[i] = 1
    else:
        predictions[i] = 0

accuracy = metrics.accuracy_score(test_set[:,-1], predictions)
precision = metrics.precision_score(test_set[:,-1], predictions)
recall = metrics.recall_score(test_set[:,-1], predictions)
confusion = metrics.confusion_matrix(test_set[:,-1], predictions)

print "Accuracy: " + str(accuracy)
print "Precision: " + str(precision)
print "Recall: " + str(recall)
print "Confusion matrix: "
print confusion
