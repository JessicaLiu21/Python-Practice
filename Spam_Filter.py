import os
import re
import mailbox
import numpy as np
import random
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

# import data from mbox files
# remove tabs, special characters and digits
# social category is labled as ham and promotions are labeled as spam

data = []
for message in mailbox.mbox('./Mail/SocialLabel.mbox'):
    subject = str(message['subject'])
    if not (('utf-8' in subject) or ('UTF-8' in subject)):
        #remove tags
        subject=re.sub("</?.*?>"," <> ", subject)

        # remove special characters and digits
        subject=re.sub("(\\d|\\W)+"," ", subject)
        
        data.append([subject, "ham", 0.0])
data = data[0:900]
print(len(data))

for message in mailbox.mbox('./Mail/PromotionsLabel.mbox'):
    subject = str(message['subject'])
    if not (('utf-8' in subject) or ('UTF-8' in subject)):
        #remove tags
        subject=re.sub("</?.*?>"," <> ", subject)

        # remove special characters and digits
        subject=re.sub("(\\d|\\W)+"," ", subject)
        
        data.append([subject, "spam", 0.0])
        
data = data[0:1800]


##KNN
# split data into train and test data
for item in data:
    item.append(random.random())
train_data = []
test_data = []
for item in data:
    if item[3] > 0.3:
        train_data.append(item[0:3])
    else:
        test_data.append(item[0:3])

# start testing
def getSimilarity(record1, record2):
    len1 = len(record1[0].split())
    len2 = len(record2[0].split())
    num_common = 0
    d = dict()
    for word in record1[0].split():
    	if word not in d:
    		d[word] = 1
    for word in record2[0].split():
    	if word in d:
    		num_common += 1
    divided_by =  (len1 * len2) ** 0.5
    if (divided_by != 0):
        similarity = num_common / divided_by
    else:
        similarity = 0
    return similarity


def findKNN(train_data, record, k):
    # get the distance between every train_data and the record
    for i in range(0,len(train_data)):
    	sim = getSimilarity(train_data[i], record)
    	train_data[i][-1] = sim

    res = []
    for i in range(k):
    	max_sim = 0
    	max_sim_index = 0
    	for i in range(0, len(train_data)):
    		if train_data[i][-1] > max_sim:
    			max_sim = train_data[i][-1]
    			max_sim_index = i
    	train_data[max_sim_index][-1] = 0
    	res.append(train_data[max_sim_index])
    return res

def calc_auc(knn):
    num_ham = 0
    num_spam = 0
    for r in knn:
        if r[1] == 'ham':
            num_ham += 1
        else:
            num_spam += 1
    pred = num_spam/(num_ham + num_spam)
    return pred

#calculate auc for k in range of 1 to 120
preds = []
aucs = []
for k in range(1,120):
    preds=[]
    for d in test_data:
        knn = findKNN(train_data, d, k)
        preds.append(calc_auc(knn))       
    TPRs = []
    FPRs = []
    preds.sort()
    for threshold in preds:
        TN = 0
        TP = 0
        FP = 0
        FN = 0
        for p, d in zip(preds, test_data):
            if p >= threshold:
                result = "spam"
            else:
                result = "ham"

            #calculating metrics
            if result == d[1]:
                if result == 'spam':
                    TP += 1
                if result == 'ham':
                    TN += 1
            else:
                if result == 'spam':
                    FP += 1
                if result == 'ham':
                    FN += 1
        TPR = TP/(TP+FN)
        FPR = FP/(FP+TN)
        TPRs.append(TPR)
        FPRs.append(FPR)
    auc = metrics.auc(FPRs, TPRs)
    aucs.append(auc) 

plt.plot(range(1,120), aucs)
plt.show()

def judge(knn):
    num_ham = 0
    num_spam = 0
    for r in knn:
        if r[1] == 'ham':
            num_ham += 1
        else:
            num_spam += 1
    p_spam = num_spam/(num_ham + num_spam)
    return "spam" if p_spam >= 0.36 else "ham"

#final calculation for KNN after deciding k=11
k=11
TN = 0
TP = 0
FP = 0
FN = 0
aucs = []
preds=[]
for d in test_data:
    knn = findKNN(train_data, d, k)
    preds.append(calc_auc(knn))
    result = judge(knn)
    if result == d[1]:
        correct += 1
        if result == 'spam':
            TP += 1
        if result == 'ham':
            TN += 1
    else:
        wrong += 1
        if result == 'spam':
            FP += 1
        if result == 'ham':
            FN += 1

## NAIVE BAYES
# map tokens to number of occurences
ham_dict = dict()
spam_dict = dict()
for d in train_data:
	if d[-1] == "ham":
		for word in d[0].split():
			if word in ham_dict:
				ham_dict[word] += 1
			else:
				ham_dict[word] = 1
	elif d[-1] == "spam":
		for word in d[0].split():
			if word in spam_dict:
				spam_dict[word] += 1
			else:
				spam_dict[word] = 1

# testing
prior_ham = 0.5
prior_spam = 0.5
TN = 0
TP = 0
FP = 0
FN = 0
pred = []
for d in test_data:
	text = d[0]
	p_ham = 1
	p_spam = 1
	for word in text.split():
		num_ham = ham_dict[word] if word in ham_dict else 0.000001
		num_spam = spam_dict[word] if word in spam_dict else 0.000001
		likily_ham = num_ham / (num_ham + num_spam)
		likily_spam = num_spam / (num_ham + num_spam)
		p_ham *= (likily_ham * prior_ham) / (likily_ham * prior_ham + likily_spam * prior_spam)
		p_spam *= (likily_spam * prior_spam) / (likily_ham * prior_ham + likily_spam * prior_spam)
	if p_spam > p_ham:
		result = "spam"
	elif p_spam < p_ham:
		result = "ham"
	p_spam_roc = p_spam/(p_spam+p_ham)
	pred.append(p_spam_roc)

#calculating metrics
	if result == d[1]:
		if result == 'spam':
			TP += 1
		if result == 'ham':
			TN += 1
	else:
		if result == 'spam':
			FP += 1
		if result == 'ham':
			FN += 1
precision=TP/(TP+FP)
recall=TP/(TP+FN)
F1 = 2 * (precision * recall) / (precision + recall)


#calculating AUC for Naive Bayes
TPRs = []
FPRs = []
pred.sort()
for threshold in pred:
    TN = 0
    TP = 0
    FP = 0
    FN = 0
    for p, d in zip(pred, test_data):
        if p > threshold:
            result = "spam"
        else:
            result = "ham"

        #calculating metrics
        if result == d[1]:
            if result == 'spam':
                TP += 1
            if result == 'ham':
                TN += 1
        else:
            if result == 'spam':
                FP += 1
            if result == 'ham':
                FN += 1

    TPR = TP/(TP+FN)
    FPR = FP/(FP+TN)
    TPRs.append(TPR)
    FPRs.append(FPR)