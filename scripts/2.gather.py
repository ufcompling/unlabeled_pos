### Get all results, including results for analyses of individual tags

### python scripts/2.gather.py 500

### Why different POS tags show different learning curves?
### Conjecture 1: the probability of each POS tag in the training data
### Conjecture 2: the entropy of words for each POS tag


import io, os, sys
import numpy as np
from scipy.stats import entropy
import nltk

base = 2  # work in units of bits

datadir = '/orange/ufdatastudios/zoey.liu//unlabeled_pos/pos_data/'

select_interval = '500' #sys.argv[1]

sizes = []
sizes.append(sys.argv[1])

task = 'pos'
ud_tags = ['PRON', 'SCONJ', 'PROPN', 'VERB', 'ADP', 'PUNCT', 'NOUN', 'CCONJ', 'ADV', 'DET', 'ADJ', 'AUX', 'PART', 'NUM', 'SYM', 'INTJ', 'X']

# Read CoNLL-U format
def conll_read_sentence(file_handle):

	sent = []

	for line in file_handle:
		line = line.strip('\n')
		if line.startswith('#') is False:
			toks = line.split("\t")
			if len(toks) != 10 and sent not in [[], ['']] and list(set([tok[1] for tok in sent])) != ['_']:
				return sent 
			if len(toks) == 10 and '-' not in toks[0] and '.' not in toks[0]:
				if toks[0] == 'q1':
					toks[0] = '1'
				sent.append(toks)

	toks = [tok[1] for tok in sent]
	lemmas = [tok[2] for tok in sent]
	if (set(toks) == '_' and set(lemmas) == '_') or len(sent) == 0: ## filter out treebanks where the data is empty without needing special access
		return None

	return None

## Collect data from the training or the test set
def collect_data(file_handle):
	data = []
	with open(file_handle, encoding = 'utf-8') as f:
		sent = conll_read_sentence(f)
		while sent is not None:
			words = []
			for tok in sent:
				temp_word = tok[1]
				word = ''
				if ' ' in temp_word:
					word = ''.join(t for t in temp_word.split())
				else:
					word = temp_word
				words.append(word)
			POS_tags = [tok[3] for tok in sent]

			filter_count = 0
			if len(words) == 1: # sentence with one word
				filter_count += 1
			if len(words) == 2 and (POS_tags[0] == 'PUNCT' or POS_tags[1] == 'PUNCT'): # sentence with two words one of which is punctuation
				filter_count += 1
			if list(set(POS_tags)) == ['PUNCT']:
				filter_count += 1
			http_count = 0
			for w in words:
				w = w.lower()
				if w.startswith('http'):
					http_count += 1
			if http_count != 0: # sentence that contains url
				filter_count += 1

			if filter_count == 0:
				if [words, POS_tags] not in data:
					data.append([words, POS_tags])
			sent = conll_read_sentence(f)

	return data

## Calculate the number of sentences and tokens in a training or test set
def descriptive(data):
	n_sents = len(data)
	n_toks = sum([len(tok[0]) for tok in data])
	return n_sents, n_toks


### Calculate the frequency and probability of each tag
def tag_stats(tag_list):
	tag_stats_dict = {}
	total = len(tag_list)
	for tag in ud_tags:
		tag_freq = tag_list.count(tag)
		tag_stats_dict[tag] = [tag_freq]

	for tag in ud_tags:
		tag_prob = tag_stats_dict[tag][0] / total
		tag_stats_dict[tag].append(tag_prob)

	return tag_stats_dict


### Given a list of words, calculate entropy
def word_entropy(word_list):
	total = len(word_list)
	word_prob_list = []
	for w in set(word_list):
		word_prob_list.append(word_list.count(w) / total)

	word_prob_list = np.array(word_prob_list)
	H = entropy(word_prob_list, base = base)

	return H


### Given a training input and an output file, get the list of words for each tag
def tag_word_dist(train_input, train_output):
	train_input_data = []
	with open(train_input) as f:
		for line in f:
			train_input_data.append(line.strip().split())

	train_output_data = []
	with open(train_output) as f:
		for line in f:
			train_output_data.append(line.strip().split())

	tag_word_dict = {}
	for tag in ud_tags:
		tag_word_dict[tag] = []

	for i in range(len(train_input_data)):
		sent = train_input_data[i]
		tags = train_output_data[i]
		for z in range(len(sent)):
			w = sent[z]
			tag = tags[z]
			tag_word_dict[tag].append(w)

	new_tag_word_dict = {}
	for tag, word_list in tag_word_dict.items():
		if tag_word_dict[tag] != []:
			new_tag_word_dict[tag] = word_entropy(tag_word_dict[tag])
		else:
			new_tag_word_dict[tag] = 'None'

	return new_tag_word_dict


### Calcualte the entropy of the bigram distributions for a given tag
def tag_syntax_dist(train_bigram_list, uni_tag):
	tag_syntax_env = []
	for bigram in train_bigram_list:
		if uni_tag in bigram:
			# the tag only occurs once in the bigram
			if bigram.count(uni_tag) == 1:
				for gram in bigram:
					if gram != tag:
						tag_syntax_env.append(gram)
			else:
				tag_syntax_env.append(uni_tag)

	return word_entropy(tag_syntax_env)

os.makedirs('pos_results/', exist_ok = True)

for size in sizes:
	for treebank in os.listdir('pos_data/'):
		if os.path.exists(datadir + treebank):
			treebank_outfile_path = 'pos_results/' + treebank + '_' + size + '_' + task + '_results_new.txt'
			treebank_outfile = None
			if not os.path.exists(treebank_outfile_path) or os.path.getsize(treebank_outfile_path) == 0:
				print(treebank)
				treebank_outfile = open('pos_results/' + treebank + '_' + size + '_' + task + '_results_new.txt', 'w')
				treebank_header = ['Treebank', 'Task', 'Size', 'Select_interval', 'Select_size', 'Method', 'Total', 'Model', 'Metric', 'Value', 'Seed']
				treebank_outfile.write(' '.join(tok for tok in treebank_header) + '\n')

			treebank_individual_outfile_path = 'pos_results/' + treebank + '_' + size + '_' + task + '_results_inidvidual_tag_new.txt'
			treebank_individual_outfile = None
			if not os.path.exists(treebank_individual_outfile_path) or os.path.getsize(treebank_individual_outfile_path) == 0:
				treebank_individual_outfile = open('pos_results/' + treebank + '_' + size + '_' + task + '_results_inidvidual_tag_new.txt', 'w')
				treebank_individual_header = ['Treebank', 'Task', 'Size', 'Select_interval', 'Select_size', 'Method', 'Total', 'Tag', 'Model', 'Metric', 'Value', 'Support', 'Tag_Prob', 'Tag_word_entropy', 'Tag_syntax_entropy', 'Seed']
				treebank_individual_outfile.write(' '.join(tok for tok in treebank_individual_header) + '\n')

			train_file = ''
			for file in os.listdir('/blue/liu.ying/unlabeled_pos/ud-treebanks-v2.14/' + treebank + '/'):
				if 'train.conllu' in file:
					train_file = '/blue/liu.ying/unlabeled_pos/ud-treebanks-v2.14/' + treebank + '/' + file

			train_data = collect_data(train_file)
			train_n_sents, train_n_toks = descriptive(train_data)

			max_size = 0
			if train_n_toks <= 100000:
				max_size = train_n_toks
			else:
				max_size = 100000

			if size in os.listdir(datadir + treebank):
				for arch in ['crf']: #, 'transformer_tiny', 'lstm']:					
					for method in ['al', 'random']:
						iterations = int(max_size / int(select_interval))
						for i in range(iterations):
							total_list = []
							precision_scores = []
							recall_scores = []
							f1_scores = []
							for z in range(3):
								select = str(i * int(select_interval))
								train_input = datadir + treebank + '/' + size + '/' + str(z) + '/' + method + '/' + select_interval + '/select' + str(select) + '/train.' + size + '.input'
								kl_file = datadir + treebank + '/' + size + '/' + str(z) + '/' + method + '/' + select_interval + '/select' + str(select) + '/' + arch + '/kl.txt'
								classification_report = datadir + treebank + '/' + size + '/' + str(z) + '/' + method + '/' + select_interval + '/select' + str(select) + '/' + arch + '/classification_report.txt'
								

								if treebank_outfile is not None and treebank_individual_outfile is not None and os.path.exists(train_input) is True and os.path.exists(kl_file) is True and os.path.exists(classification_report) is True and int(size) + int(select) <= max_size:
									total = 0
									select = str(i * int(select_interval))
									train_input = datadir + treebank + '/' + size + '/' + str(z) + '/' + method + '/' + select_interval + '/select' + str(select) + '/train.' + size + '.input'									
									with open(train_input) as f:
										for train_line in f:
											toks = train_line.strip().split()
											total += len(toks)
									total_list.append(total)
									select = str(i * int(select_interval))
									

									evaluation_file = datadir + treebank + '/' + size + '/' + str(z) + '/' + method + '/' + select_interval + '/select' + str(select) + '/' + arch + '/eval.txt'
									if int(size) + int(select) <= max_size and os.path.exists(evaluation_file):
										precision = ''
										recall = ''
										f1 = ''
										with open(evaluation_file) as f:
											for line in f:
												toks = line.strip().split()
												if line.startswith('Precision'):
													precision = toks[1]
												if line.startswith('Recall'):
													recall = toks[1]
												if line.startswith('F1'):
													f1 = toks[1]

										precision_scores.append(float(precision))
										recall_scores.append(float(recall))
										f1_scores.append(float(f1))

							if len(precision_scores) == 3:
							### Adding average evaluation metrics
								ave_info = [treebank, task, size, select_interval, select, method, int(np.mean(total)), arch, 'precision', np.mean(precision_scores), 'average']
								treebank_outfile.write(' '.join(str(tok) for tok in ave_info) + '\n')
								ave_info = [treebank, task, size, select_interval, select, method, int(np.mean(total)), arch, 'recall', np.mean(precision_scores), 'average']
								treebank_outfile.write(' '.join(str(tok) for tok in ave_info) + '\n')
								ave_info = [treebank, task, size, select_interval, select, method, int(np.mean(total)), arch, 'f1', np.mean(precision_scores), 'average']
								treebank_outfile.write(' '.join(str(tok) for tok in ave_info) + '\n')

								std_info = [treebank, task, size, select_interval, select, method, int(np.mean(total)), arch, 'precision', np.std(precision_scores), 'std']
								treebank_outfile.write(' '.join(str(tok) for tok in std_info) + '\n')
								std_info = [treebank, task, size, select_interval, select, method, int(np.mean(total)), arch, 'recall', np.std(precision_scores), 'std']
								treebank_outfile.write(' '.join(str(tok) for tok in std_info) + '\n')
								std_info = [treebank, task, size, select_interval, select, method, int(np.mean(total)), arch, 'f1', np.std(precision_scores), 'std']
								treebank_outfile.write(' '.join(str(tok) for tok in std_info) + '\n')

