from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support
import sklearn_crfsuite
import matplotlib.pyplot as plt
import string
import itertools
import datetime
import re
import pickle
import io, os, sys, subprocess
import statistics

datadir = sys.argv[1]
treebank = sys.argv[2]
size = sys.argv[3]
select_interval = sys.argv[4]
select = sys.argv[5]
arch = sys.argv[6]
method = sys.argv[7]

sub_datadir = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/'

subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size])
subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size + '/' + method])
subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size + '/' + method + '/' + select_interval])
subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select])
subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + arch])

model_dir = sub_datadir + 'crf/'

previous_datadir = ''
if select not in ['0', 'all']:
	previous_datadir = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + str(int(select) - int(select_interval)) + '/'

if select == '0':
	os.system('cp ' + datadir + treebank + '/train.' + size + '.input ' + sub_datadir)
	os.system('cp ' + datadir + treebank + '/train.' + size + '.output ' + sub_datadir)
	os.system('cp ' + datadir + treebank + '/select.' + size + '.input ' + sub_datadir)
	os.system('cp ' + datadir + treebank + '/select.' + size + '.output ' + sub_datadir)

elif select == 'all':
	os.system('cat ' + datadir + treebank + '/train.' + size + '.input ' + datadir + treebank + '/select.' + size + '.input >' + sub_datadir + 'train.' + size + '.input')
	os.system('cat ' + datadir + treebank + '/train.' + size + '.output ' + datadir + treebank + '/select.' + size + '.output >' + sub_datadir + 'train.' + size + '.output')

else:
	os.system('cat ' + previous_datadir + 'train.' + size + '.input ' + previous_datadir + '/increment.input >' + sub_datadir + 'train.' + size + '.input')
	os.system('cat ' + previous_datadir + 'train.' + size + '.output ' + previous_datadir + '/increment.output >' + sub_datadir + 'train.' + size + '.output')
	os.system('cp ' + previous_datadir + 'residual.input ' + sub_datadir + 'select.' + size + '.input')
	os.system('cp ' + previous_datadir + 'residual.output ' + sub_datadir + 'select.' + size + '.output')

### Gathering data ###

def gather_data(file):   # *.input, *.output

	data = [] # either words or labels
	with open(file) as f:
		for line in f:
			data.append(line.strip().split())

	return data


### Gathering word features ###

def word2features(sent, i):
	word = sent[i]
	features = {
		'bias': 1.0,
		'word': word,
		'len(word)': len(str(word)),
		'word[:4]': word[:4],
		'word[:3]': word[:3],
		'word[:2]': word[:2],
		'word[-3:]': word[-3:],
		'word[-2:]': word[-2:],
		'word[-4:]': word[-4:],
		'word.lower()': word.lower(),
#        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word.lower()),
		'word.ispunctuation': (word in string.punctuation),
		'word.isdigit()': word.isdigit()}
	
	if i > 0:
		word1 = sent[i-1]
		features.update({
			'-1:word': word1,
			'-1:len(word)': len(str(word)),
			'-1:word.lower()': word1.lower(),
#            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word1.lower()),
			'-1:word[:3]': word1[:3],
			'-1:word[:2]': word1[:2],
			'-1:word[-3:]': word1[-3:],
			'-1:word[-2:]': word1[-2:],
			'-1:word.isdigit()': word1.isdigit(),
			'-1:word.ispunctuation': (word1 in string.punctuation)})     
	else:
		features['BOS'] = True

	if i > 1:
		word2 = sent[i-2]
		features.update({
			'-2:word': word2,
			'-2:len(word)': len(str(word)),
			'-2:word.lower()': word2.lower(),
			'-2:word[:3]': word2[:3],
			'-2:word[:2]': word2[:2],
			'-2:word[-3:]': word2[-3:],
			'-2:word[-2:]': word2[-2:],
			'-2:word.isdigit()': word2.isdigit(),
			'-2:word.ispunctuation': (word2 in string.punctuation),
		})

	if i < len(sent)-1:
		word1 = sent[i+1]
		features.update({
			'+1:word': word1,
			'+1:len(word)': len(str(word)),
			'+1:word.lower()': word1.lower(),
			'+1:word[:3]': word1[:3],
			'+1:word[:2]': word1[:2],
			'+1:word[-3:]': word1[-3:],
			'+1:word[-2:]': word1[-2:],
			'+1:word.isdigit()': word1.isdigit(),
			'+1:word.ispunctuation': (word1 in string.punctuation),
		})

	else:
		features['EOS'] = True    
	
	if i < len(sent) - 2:
		word2 = sent[i+2]
		features.update({
			'+2:word': word2,
			'+2:len(word)': len(str(word)),
			'+2:word.lower()': word2.lower(),
#            '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)',r'\1', word2.lower()),
			'+2:word[:3]': word2[:3],
			'+2:word[:2]': word2[:2],
			'+2:word[-3:]': word2[-3:],
			'+2:word[-2:]': word2[-2:],
			'+2:word.isdigit()': word2.isdigit(),
			'+2:word.ispunctuation': (word2 in string.punctuation),
		})

	return features


# Convert each sentence to a list of features 
def sent2features(sent):
#	print(sent)
	return [word2features(sent, i) for i in range(len(sent))]


# Construct a feature vector
def feature_vectors(input_file, output_file):

	sent_list = gather_data(input_file)
	pos_list = gather_data(output_file)

	X = [] # list (learning set) of list (word) of dics (chars), INPUT for crf
	Y = [] # list (learning set) of list (word) of labels (chars), INPUT for crf
	
	X = [sent2features(sent) for sent in sent_list]
	Y = pos_list

	return X, Y


# Build a confusion matrix
def cm(gold_tags, predicted_tags, taglist, modelname):
	'''builds and display a confusion matrix so we 
	can evaluate where our tagger is making wrong 
	predictions, after we test a POS tagger'''
	
	alpha_taglist = sorted(set(taglist))
	confusion_matrix = metrics.confusion_matrix(gold_tags,predicted_tags,labels=alpha_taglist,normalize="true")
	disp = metrics.ConfusionMatrixDisplay(confusion_matrix,display_labels=alpha_taglist)

	plt.rcParams["figure.figsize"] = (17,17)
	plt.rcParams.update({'font.size': 12})
	disp.plot(colorbar=False)
#	plt.show() # display below
	#save as file
	plt.savefig(model_dir + '_matrix.png')    
	
	matrixstring = '{0:5s}'.format(' ') + '\t'.join(['{0:^4s}'.format(tag) for tag in alpha_taglist]) + '\n'
	for i,row in enumerate(confusion_matrix):
		cols = '\t'.join(['{:.2f}'.format(round(col,2)) for col in row])
		matrixstring+='{0:6s}'.format(alpha_taglist[i]) + cols + '\n'
	
	return matrixstring

	
def logResults(testgold, testpredict, confusionreport, modelname, tagset):
	time = str(datetime.datetime.now())
	testgold = list(itertools.chain.from_iterable(testgold))
	testpredict = list(itertools.chain.from_iterable(testpredict))
	classreport = metrics.classification_report(testgold, testpredict, target_names = tagset) #zero_division=0.0)
	report = '\n\nClassification Report\n\n{}\n\nConfusion Matrix\n\n{}\n'.format(classreport, confusionreport)
	with open(model_dir + 'classification_report.txt', 'w') as R:
		R.write(time + report)
		

def datafile(filename, data):
    with open(filename, 'w') as T:
        T.write('\n'.join(data))


def sort_confidence(selection_sents, selection_tags, confscores, modelname):
	'''sort auto-annotated sentences based on how "confident" 
	the model was at it predictions of each sentence's POS tags, 
	by increasing "confidence", 
	lower probability == less confidence.
	Writes to file.'''
	
	SEPARATOR = '|'

	with_confidence = list(zip(selection_sents, selection_tags, confscores))
	with_confidence.sort(key = lambda x: x[2])
	sorted_sents = [z[0] for z in with_confidence]
	sorted_tags = [z[1] for z in with_confidence]
	sorted_confidence = [z[2] for z in with_confidence]

	datastring = []
	for i in range(len(sorted_sents)):
		sent = sorted_sents[i]
		tags = sorted_tags[i]
		pairs = list(zip(sent, tags))
		info = ' '.join([pair[0] + SEPARATOR + pair[1] for pair in pairs]) + '\t' + str(sorted_confidence[i])
		datastring.append(info)

	datafile(model_dir + 'confidence.predict', datastring)

	return with_confidence


### Building models ###

def mainCRF():

	test_input = datadir + treebank + '/test.full.input'
	test_output = datadir + treebank + '/test.full.output'

	train_input = sub_datadir + 'train.' + size + '.input'
	train_output = sub_datadir + 'train.' + size + '.output'
	select_input = sub_datadir + 'select.' + size + '.input'
	select_output = sub_datadir + 'select.' + size + '.output'

	# get tag set
	tagset = []
	train_tags = gather_data(train_output)
	for tag_list in train_tags:
		tagset += tag_list
	n_train_toks = len(tagset)
	n_train_sents = len(train_tags)
	tagset = list(set(tagset))
#	print(tagset)

	train_X, train_Y = feature_vectors(train_input, train_output)
	test_X, test_Y = feature_vectors(test_input, test_output)
	if select != 'all':
		select_X, select_Y = feature_vectors(select_input, select_output)
		select_sents = gather_data(select_input)
		select_tags = gather_data(select_output)

	# training parameters
	crf = sklearn_crfsuite.CRF(
		algorithm = 'lbfgs',
		c1 = 0.25,
		c2 = 0.3,
		max_iterations = 100,
		all_possible_transitions=True)

	for i in range(len(train_X)):
		if len(train_X[i]) != len(train_Y[i]):			
			sent = []
			for tok in train_X[i]:
				sent.append(tok['word'])
			print(' '.join(w) for w in sent)
			print(train_input)
			print(train_Y[i])
			print('\n')
	#train
	crf.fit(train_X, train_Y)

	# save model
	pickle.dump(crf, io.open(model_dir + 'model.pkl', "wb"))
	print('model saved')

	# testing
	test_output = crf.predict(test_X)

	# get test reports
	all_test_labels = []
	for tag_list in test_Y:
		all_test_labels += tag_list

	all_test_preds = []
	for tag_list in test_output:
		all_test_preds += tag_list

	print(len(all_test_labels), len(all_test_preds))

#	matrix = cm(all_test_labels, all_test_preds, tagset, 'CRF')
#	logResults(test_Y, test_output, matrix, 'CRF', tagset) # would not work if the numbers of tagset in the training and test sets are different
#	print('confusion matrix built')

	# output overall evaluation metrics
	evaluation_file = model_dir + 'eval.txt'
	precision_scores = []
	recall_scores = []
	f1_scores = []
	for i in range(len(test_Y)):
		y_true = test_Y[i]
		y_pred = test_output[i]
		scores = precision_recall_fscore_support(y_true, y_pred, average='weighted')
		precision, recall, f1 = scores[0], scores[1], scores[2]
		precision_scores.append(precision)
		recall_scores.append(recall)
		f1_scores.append(f1)

	average_precision = statistics.mean(precision_scores)
	average_recall = statistics.mean(recall_scores)
	average_f1 = statistics.mean(f1_scores)

	with open(evaluation_file, 'w') as f:
		f.write('Precision: ' + str(average_precision) + '\n')
		f.write('Recall: ' + str(average_recall) + '\n')
		f.write('F1: ' + str(average_f1) + '\n')
		f.write('N of training tokens: ' + str(n_train_toks) + '\n')
		f.write('N of training sents: ' + str(n_train_sents) + '\n')

	print('eval file generated')
	print(treebank, size, average_precision, average_recall, average_f1)

	# predict on selection file
	if select != 'all':
		predicted_labels = crf.predict(select_X)
		predicted_sequences = [list(zip(select_sents[i], predicted_labels[i])) for i in range(len(predicted_labels))]

		# get confidence score
		all_probs = crf.predict_marginals(select_X)
		confidences = []
		for s,sent in enumerate(predicted_sequences):
			confidences.append(sum(all_probs[s][i][wordpair[1]] for i, wordpair in enumerate(sent))/len(sent))

		confidence_data = sort_confidence(select_sents, select_tags, confidences, 'CRF')
		print('selection predictions generated')

		# generate increment.input and residual.input
		selection_sorted_sents = [z[0] for z in confidence_data]
		selection_sorted_tags = [z[1] for z in confidence_data]
		selection_sorted_confidence = [z[2] for z in confidence_data]

		increment_input = open(sub_datadir + 'increment.input', 'w')
		increment_output = open(sub_datadir + 'increment.output', 'w')
		n_toks = 0
		i = 0
		while n_toks <= int(select_interval):
			sent = selection_sorted_sents[i]
			tags = selection_sorted_tags[i]
			increment_input.write(' '.join(w for w in sent) + '\n')
			increment_output.write(' '.join(w for w in tags) + '\n')
			n_toks += len(sent)
			i += 1

		residual_sents = selection_sorted_sents[i : ]
		residual_tags = selection_sorted_tags[i: ]
		residual_input = open(sub_datadir + 'residual.input', 'w')
		residual_output = open(sub_datadir + 'residual.output', 'w')
		for i in range(len(residual_sents)):
			sent = residual_sents[i]
			tags = residual_tags[i]
			residual_input.write(' '.join(w for w in sent) + '\n')
			residual_output.write(' '.join(w for w in tags) + '\n')

mainCRF()
