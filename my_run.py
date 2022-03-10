# chang's run.py
from utils import *
import pprint
from collections import Counter
import math

def naive_bayes(data_percentage, smooth=True, log_likelihood=True):
	
	percentage_positive_instances_train = data_percentage
	percentage_negative_instances_train = data_percentage

	percentage_positive_instances_test  = data_percentage
	percentage_negative_instances_test  = data_percentage
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	# print("Number of positive training instances:", len(pos_train))
	# print("Number of negative training instances:", len(neg_train))
	# print("Number of positive test instances:", len(pos_test))
	# print("Number of negative test instances:", len(neg_test))

	with open('vocab.txt','w') as f:
		for word in vocab:
			f.write("%s\n" % word)
	# print("Vocabulary (training set):", len(vocab))

	vocab_size = len(vocab)
	# Calculate the prior probabilities
	prior_pos = len(pos_train) / (len(pos_train) + len(neg_train))
	prior_neg = len(neg_train) / (len(pos_train) + len(neg_train))

	# print("Prior probability of positive class:", prior_pos)
	# print("Prior probability of negative class:", prior_neg)

	# Build the likelihoods table
	train_dict = {}
	for word in vocab:
		train_dict[word] = 0;
	
	likelihoods = {}
	likelihoods["pos"] = train_dict.copy()
	likelihoods["pos"].update( dict(Counter(sum(pos_train, []))) )
	likelihoods["neg"] = train_dict.copy()
	likelihoods["neg"].update( dict(Counter(sum(neg_train, []))) )

	word_count_pos = sum(likelihoods["pos"].values()) 
	word_count_neg = sum(likelihoods["neg"].values())

	model_pos = {}
	model_neg = {}

	# calculate probablity, apply lapalce smoothing 
	for word in likelihoods["pos"]:
		if smooth:
			model_pos[word] = (likelihoods["pos"][word] + 1) / (word_count_pos + vocab_size) 
		else:
			model_pos[word] = likelihoods["pos"][word] / word_count_pos

	for word in likelihoods["neg"]:
		if smooth:
			model_neg[word] = (likelihoods["neg"][word] + 1) / (word_count_neg + vocab_size)
		else:
			model_neg[word] = likelihoods["neg"][word] / word_count_neg

	pos_test_correct = 0
	for doc in pos_test:
		doc_dict = dict(Counter(doc))
		doc_p_pos = math.log(prior_pos) if log_likelihood else prior_pos
		doc_p_neg = math.log(prior_neg) if log_likelihood else prior_neg
		for word in doc_dict:
			if word in model_pos: 
				# it should also exist in the negative vacabulary
				if log_likelihood:
					doc_p_pos += math.log(model_pos[word]) if model_pos[word] != 0 else 0
					doc_p_neg += math.log(model_neg[word]) if model_neg[word] != 0 else 0
				else: 
					doc_p_pos *= model_pos[word]
					doc_p_neg *= model_neg[word]
		if (doc_p_pos > doc_p_neg):
			pos_test_correct += 1

	neg_test_correct = 0
	for doc in neg_test:
		doc_dict = dict(Counter(doc))
		doc_p_pos = math.log(prior_pos) if log_likelihood else prior_pos
		doc_p_neg = math.log(prior_neg) if log_likelihood else prior_neg
		for word in doc_dict:
			if word in model_pos:
				# it should also exist in the negative vacabulary
				if log_likelihood:
					doc_p_pos += math.log(model_pos[word]) if model_pos[word] != 0 else 0
					doc_p_neg += math.log(model_neg[word]) if model_neg[word] != 0 else 0
				else:
					doc_p_pos *= model_pos[word]
					doc_p_neg *= model_neg[word]
		if (doc_p_pos < doc_p_neg):
			neg_test_correct += 1

	# print("correct Pos Test: ", pos_test_correct);
	# print("correct Neg Test: ", neg_test_correct);
	
	accuracy = (pos_test_correct + neg_test_correct) / (len(pos_test) + len(neg_test))
	precision = pos_test_correct / (pos_test_correct + len(neg_test) - neg_test_correct)
	recall = pos_test_correct / len(pos_test)
	confusion_matrix = [[pos_test_correct, len(pos_test) - pos_test_correct], [len(neg_test) - neg_test_correct, neg_test_correct]]
	return accuracy, precision, recall, confusion_matrix

(acc, pre, rec, conf) = naive_bayes(0.2, smooth=False, log_likelihood=False)
print("| **Accuracy** | **Precision** | **Recall** |")
print("| :---: | :---: | :---: |")
print("|{} | {} | {} |".format(acc, pre, rec))
print("Confusion Matrix:")
print("|  | **Predicted +** | **Predicted-** |")
print("| :--- | :--- | :--- |")
print("| **Actual +** | {} | {} |".format(conf[0][0], conf[0][1]))
print("| **Actual -** | {} | {} |".format(conf[1][0], conf[1][1]))

(acc, pre, rec, conf) = naive_bayes(0.2, smooth=False, log_likelihood=True)
print("| **Accuracy** | **Precision** | **Recall** |")
print("| :---: | :---: | :---: |")
print("|{} | {} | {} |".format(acc, pre, rec))
print("Confusion Matrix:")
print("|  | **Predicted +** | **Predicted-** |")
print("| :--- | :--- | :--- |")
print("| **Actual +** | {} | {} |".format(conf[0][0], conf[0][1]))
print("| **Actual -** | {} | {} |".format(conf[1][0], conf[1][1]))

(acc, pre, rec, conf) = naive_bayes(0.2, smooth=True, log_likelihood=True)
print("| **Accuracy** | **Precision** | **Recall** |")
print("| :---: | :---: | :---: |")
print("|{} | {} | {} |".format(acc, pre, rec))
print("Confusion Matrix:")
print("|  | **Predicted +** | **Predicted-** |")
print("| :--- | :--- | :--- |")
print("| **Actual +** | {} | {} |".format(conf[0][0], conf[0][1]))
print("| **Actual -** | {} | {} |".format(conf[1][0], conf[1][1]))

import pprint
from collections import Counter
import math

def naive_bayes_smoothing_plot(data_percentage):
	
	percentage_positive_instances_train = data_percentage
	percentage_negative_instances_train = data_percentage

	percentage_positive_instances_test  = data_percentage
	percentage_negative_instances_test  = data_percentage
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	alphas = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
	accuracies = [ deploy_nb_with_smoothing(pos_train, neg_train, vocab, pos_test, neg_test, alpha) for alpha in alphas]
	return alphas, accuracies

def deploy_nb_with_smoothing(pos_train, neg_train, vocab, pos_test, neg_test, alpha):

	vocab_size = len(vocab)
	# Calculate the prior probabilities
	prior_pos = len(pos_train) / (len(pos_train) + len(neg_train))
	prior_neg = len(neg_train) / (len(pos_train) + len(neg_train))


	train_dict = {}
	for word in vocab:
		train_dict[word] = 0;
	
	likelihoods = {}
	likelihoods["pos"] = train_dict.copy()
	likelihoods["pos"].update( dict(Counter(sum(pos_train, []))) )
	likelihoods["neg"] = train_dict.copy()
	likelihoods["neg"].update( dict(Counter(sum(neg_train, []))) )

	word_count_pos = sum(likelihoods["pos"].values()) 
	word_count_neg = sum(likelihoods["neg"].values())

	model_pos = {}
	model_neg = {}

	# calculate probablity, apply lapalce smoothing 
	for word in likelihoods["pos"]:
		model_pos[word] = (likelihoods["pos"][word] + alpha) / (word_count_pos + vocab_size*alpha) 

	for word in likelihoods["neg"]:
		model_neg[word] = (likelihoods["neg"][word] + alpha) / (word_count_neg + vocab_size*alpha)

	pos_test_correct = 0
	for doc in pos_test:
		doc_dict = dict(Counter(doc))
		doc_p_pos = math.log(prior_pos) 
		doc_p_neg = math.log(prior_neg) 
		for word in doc_dict:
			if word in model_pos: 
				# it should also exist in the negative vacabulary
				doc_p_pos += math.log(model_pos[word]) if model_pos[word] != 0 else 0
				doc_p_neg += math.log(model_neg[word]) if model_neg[word] != 0 else 0
		if (doc_p_pos > doc_p_neg):
			pos_test_correct += 1

	neg_test_correct = 0
	for doc in neg_test:
		doc_dict = dict(Counter(doc))
		doc_p_pos = math.log(prior_pos) 
		doc_p_neg = math.log(prior_neg) 
		for word in doc_dict:
			if word in model_pos:
				# it should also exist in the negative vacabulary
				doc_p_pos += math.log(model_pos[word]) if model_pos[word] != 0 else 0
				doc_p_neg += math.log(model_neg[word]) if model_neg[word] != 0 else 0
		if (doc_p_pos < doc_p_neg):
			neg_test_correct += 1

	accuracy = (pos_test_correct + neg_test_correct) / (len(pos_test) + len(neg_test))
	return accuracy

(alphas, accuracies) = naive_bayes_smoothing_plot(0.2)


import matplotlib.pyplot as plt

fig = plt.figure()
plt.plot(alphas, accuracies)
plt.xlabel('alpha')
plt.ylabel('accuracy')
plt.xscale('log')
plt.show()


# actually q_3_4
def naive_bayes_q_4_5(percentage_train, smooth_alpha, log_likelihood=True):
    
	percentage_positive_instances_train = percentage_train
	percentage_negative_instances_train = percentage_train

	percentage_positive_instances_test  = 1
	percentage_negative_instances_test  = 1
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	# print("Number of positive training instances:", len(pos_train))
	# print("Number of negative training instances:", len(neg_train))
	# print("Number of positive test instances:", len(pos_test))
	# print("Number of negative test instances:", len(neg_test))

	with open('vocab.txt','w') as f:
		for word in vocab:
			f.write("%s\n" % word)
	# print("Vocabulary (training set):", len(vocab))

	vocab_size = len(vocab)
	# Calculate the prior probabilities
	prior_pos = len(pos_train) / (len(pos_train) + len(neg_train))
	prior_neg = len(neg_train) / (len(pos_train) + len(neg_train))

	# print("Prior probability of positive class:", prior_pos)
	# print("Prior probability of negative class:", prior_neg)

	# Build the likelihoods table
	train_dict = {}
	for word in vocab:
		train_dict[word] = 0;
	
	likelihoods = {}
	likelihoods["pos"] = train_dict.copy()
	likelihoods["pos"].update( dict(Counter(sum(pos_train, []))) )
	likelihoods["neg"] = train_dict.copy()
	likelihoods["neg"].update( dict(Counter(sum(neg_train, []))) )

	word_count_pos = sum(likelihoods["pos"].values()) 
	word_count_neg = sum(likelihoods["neg"].values())

	model_pos = {}
	model_neg = {}

	# calculate probablity, apply lapalce smoothing 
	for word in likelihoods["pos"]:
		model_pos[word] = (likelihoods["pos"][word] + smooth_alpha) / (word_count_pos + vocab_size*smooth_alpha) 

	for word in likelihoods["neg"]:
		model_neg[word] = (likelihoods["neg"][word] + smooth_alpha) / (word_count_neg + vocab_size*smooth_alpha)

	pos_test_correct = 0
	for doc in pos_test:
		doc_dict = dict(Counter(doc))
		doc_p_pos = math.log(prior_pos) if log_likelihood else prior_pos
		doc_p_neg = math.log(prior_neg) if log_likelihood else prior_neg
		for word in doc_dict:
			if word in model_pos: 
				# it should also exist in the negative vacabulary
				if log_likelihood:
					doc_p_pos += math.log(model_pos[word]) if model_pos[word] != 0 else 0
					doc_p_neg += math.log(model_neg[word]) if model_neg[word] != 0 else 0
				else: 
					doc_p_pos *= model_pos[word]
					doc_p_neg *= model_neg[word]
		if (doc_p_pos > doc_p_neg):
			pos_test_correct += 1

	neg_test_correct = 0
	for doc in neg_test:
		doc_dict = dict(Counter(doc))
		doc_p_pos = math.log(prior_pos) if log_likelihood else prior_pos
		doc_p_neg = math.log(prior_neg) if log_likelihood else prior_neg
		for word in doc_dict:
			if word in model_pos:
				# it should also exist in the negative vacabulary
				if log_likelihood:
					doc_p_pos += math.log(model_pos[word]) if model_pos[word] != 0 else 0
					doc_p_neg += math.log(model_neg[word]) if model_neg[word] != 0 else 0
				else:
					doc_p_pos *= model_pos[word]
					doc_p_neg *= model_neg[word]
		if (doc_p_pos < doc_p_neg):
			neg_test_correct += 1

	# print("correct Pos Test: ", pos_test_correct);
	# print("correct Neg Test: ", neg_test_correct);
	
	accuracy = (pos_test_correct + neg_test_correct) / (len(pos_test) + len(neg_test))
	precision = pos_test_correct / (pos_test_correct + len(neg_test) - neg_test_correct)
	recall = pos_test_correct / len(pos_test)
	confusion_matrix = [[pos_test_correct, len(pos_test) - pos_test_correct], [len(neg_test) - neg_test_correct, neg_test_correct]]
	return accuracy, precision, recall, confusion_matrix

(acc, pre, rec, conf) = naive_bayes_q_4_5(1, 10, log_likelihood=True)
print("| **Accuracy** | **Precision** | **Recall** |")
print("| :---: | :---: | :---: |")
print("|{} | {} | {} |".format(acc, pre, rec))
print("Confusion Matrix:")
print("|  | **Predicted +** | **Predicted-** |")
print("| :--- | :--- | :--- |")
print("| **Actual +** | {} | {} |".format(conf[0][0], conf[0][1]))
print("| **Actual -** | {} | {} |".format(conf[1][0], conf[1][1]))

(acc, pre, rec, conf) = naive_bayes_q_4_5(0.5, 10, log_likelihood=True)
print("| **Accuracy** | **Precision** | **Recall** |")
print("| :---: | :---: | :---: |")
print("|{} | {} | {} |".format(acc, pre, rec))
print("Confusion Matrix:")
print("|  | **Predicted +** | **Predicted-** |")
print("| :--- | :--- | :--- |")
print("| **Actual +** | {} | {} |".format(conf[0][0], conf[0][1]))
print("| **Actual -** | {} | {} |".format(conf[1][0], conf[1][1]))

# actually q_3_4
def naive_bayes_unbalance(percentage_positive_instances_train, percentage_negative_instances_train, smooth_alpha, log_likelihood=True):
    
	percentage_positive_instances_test  = 1
	percentage_negative_instances_test  = 1
	
	(pos_train, neg_train, vocab) = load_training_set(percentage_positive_instances_train, percentage_negative_instances_train)
	(pos_test,  neg_test)         = load_test_set(percentage_positive_instances_test, percentage_negative_instances_test)

	# print("Number of positive training instances:", len(pos_train))
	# print("Number of negative training instances:", len(neg_train))
	# print("Number of positive test instances:", len(pos_test))
	# print("Number of negative test instances:", len(neg_test))

	with open('vocab.txt','w') as f:
		for word in vocab:
			f.write("%s\n" % word)
	# print("Vocabulary (training set):", len(vocab))

	vocab_size = len(vocab)
	# Calculate the prior probabilities
	prior_pos = len(pos_train) / (len(pos_train) + len(neg_train))
	prior_neg = len(neg_train) / (len(pos_train) + len(neg_train))

	# print("Prior probability of positive class:", prior_pos)
	# print("Prior probability of negative class:", prior_neg)

	# Build the likelihoods table
	train_dict = {}
	for word in vocab:
		train_dict[word] = 0;
	
	likelihoods = {}
	likelihoods["pos"] = train_dict.copy()
	likelihoods["pos"].update( dict(Counter(sum(pos_train, []))) )
	likelihoods["neg"] = train_dict.copy()
	likelihoods["neg"].update( dict(Counter(sum(neg_train, []))) )

	word_count_pos = sum(likelihoods["pos"].values()) 
	word_count_neg = sum(likelihoods["neg"].values())

	model_pos = {}
	model_neg = {}

	# calculate probablity, apply lapalce smoothing 
	for word in likelihoods["pos"]:
		model_pos[word] = (likelihoods["pos"][word] + smooth_alpha) / (word_count_pos + vocab_size*smooth_alpha) 

	for word in likelihoods["neg"]:
		model_neg[word] = (likelihoods["neg"][word] + smooth_alpha) / (word_count_neg + vocab_size*smooth_alpha)

	pos_test_correct = 0
	for doc in pos_test:
		doc_dict = dict(Counter(doc))
		doc_p_pos = math.log(prior_pos) if log_likelihood else prior_pos
		doc_p_neg = math.log(prior_neg) if log_likelihood else prior_neg
		for word in doc_dict:
			if word in model_pos: 
				# it should also exist in the negative vacabulary
				if log_likelihood:
					doc_p_pos += math.log(model_pos[word]) if model_pos[word] != 0 else 0
					doc_p_neg += math.log(model_neg[word]) if model_neg[word] != 0 else 0
				else: 
					doc_p_pos *= model_pos[word]
					doc_p_neg *= model_neg[word]
		if (doc_p_pos > doc_p_neg):
			pos_test_correct += 1

	neg_test_correct = 0
	for doc in neg_test:
		doc_dict = dict(Counter(doc))
		doc_p_pos = math.log(prior_pos) if log_likelihood else prior_pos
		doc_p_neg = math.log(prior_neg) if log_likelihood else prior_neg
		for word in doc_dict:
			if word in model_pos:
				# it should also exist in the negative vacabulary
				if log_likelihood:
					doc_p_pos += math.log(model_pos[word]) if model_pos[word] != 0 else 0
					doc_p_neg += math.log(model_neg[word]) if model_neg[word] != 0 else 0
				else:
					doc_p_pos *= model_pos[word]
					doc_p_neg *= model_neg[word]
		if (doc_p_pos < doc_p_neg):
			neg_test_correct += 1

	# print("correct Pos Test: ", pos_test_correct);
	# print("correct Neg Test: ", neg_test_correct);
	
	accuracy = (pos_test_correct + neg_test_correct) / (len(pos_test) + len(neg_test))
	precision = pos_test_correct / (pos_test_correct + len(neg_test) - neg_test_correct)
	recall = pos_test_correct / len(pos_test)
	confusion_matrix = [[pos_test_correct, len(pos_test) - pos_test_correct], [len(neg_test) - neg_test_correct, neg_test_correct]]
	return accuracy, precision, recall, confusion_matrix