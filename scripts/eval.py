import sys
from sklearn.metrics import precision_recall_fscore_support
import statistics

def metrics(gold_word, pred_word):

	correct_total = 0

	for m in pred_word:
		if m in gold_word:
			correct_total += 1

	gold_total = len(gold_word)
	pred_total = len(pred_word)

	precision = correct_total / pred_total
	recall = correct_total / gold_total

	F1 = 0

	try:
		F1 = 2 * (precision * recall) / (precision + recall)
		F1 = round(F1, 2)
	except:
		F1 = 0

	return round(precision, 2), round(recall, 2), F1

datadir = sys.argv[1]
treebank = sys.argv[2]
size = sys.argv[3]
select_interval = sys.argv[4]
select = sys.argv[5]
arch = sys.argv[6]
method = sys.argv[7]

TESTIN_gold = datadir + treebank + '/test.full.output'

### Collecting test set gold data
test_gold = []
with open(TESTIN_gold) as f:
	for line in f:
		toks = line.strip().split()
		test_gold.append(toks)

print(len(test_gold))
print('\n')

precision_scores = []
recall_scores = []
f1_scores = []

for seed in ['1', '2', '3']:
	seed_precision_scores = []
	seed_recall_scores = []
	seed_f1_scores = []

	PREDDIR = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + arch + '/' + seed + '/preds/'
	GUESSPATH = PREDDIR + treebank + '_' + size + '.testpredict'
	print(seed, GUESSPATH)

	## Extracting predictions
	test_predictions = []
	with open(GUESSPATH) as f:
		for line in f:
			toks = line.strip().split()
			test_predictions.append(toks)

	for i in range(len(test_gold)):
		y_true = test_gold[i]
		y_pred = test_predictions[i]
	#	print(y_true, '\t', y_pred)
		scores = precision_recall_fscore_support(y_true, y_pred, average='weighted')
	#	precision, recall, f1 = metrics(y_true, y_pred)
		precision, recall, f1 = scores[0], scores[1], scores[2]
		seed_precision_scores.append(precision)
		seed_recall_scores.append(recall)
		seed_f1_scores.append(f1)
		print(precision, recall, f1)
	seed_average_precision = statistics.mean(seed_precision_scores)
	seed_average_recall = statistics.mean(seed_recall_scores)
	seed_average_f1 = statistics.mean(seed_f1_scores)

	precision_scores.append(seed_average_precision)
	recall_scores.append(seed_average_recall)
	f1_scores.append(seed_average_f1)

### Average evaluation metrics
average_precision = statistics.mean(precision_scores)
average_recall = statistics.mean(recall_scores)
average_f1 = statistics.mean(f1_scores)
evaluation_file = open(datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + arch + '/eval.txt', 'w')
evaluation_file.write('Precision: ' + str(average_precision) + '\n')
evaluation_file.write('Recall: ' + str(average_recall) + '\n')
evaluation_file.write('F1: ' + str(average_f1) + '\n')

print('Precision: ' + str(average_precision) + '\n')
print('Recall: ' + str(average_recall) + '\n')
print('F1: ' + str(average_f1) + '\n')

