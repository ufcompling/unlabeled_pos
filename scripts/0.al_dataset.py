# To run,
# python scripts/0.al_dataset.py 500
# 500 is the initial training set size

# curl --remote-name-all https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-5502{/ud-treebanks-v2.14.tgz,/ud-documentation-v2.14.tgz,/ud-tools-v2.14.tgz}

import io, os, sys
import random

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

#	toks = [tok[1] for tok in sent]
#	lemmas = [tok[2] for tok in sent]
#	if (set(toks) == '_' and set(lemmas) == '_') or len(sent) == 0: ## filter out treebanks where the data is empty without needing special access
#		return None

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
		#	if len(words) == 1 and POS_tags[0] == 'PUNCT':
		#		pass
		#	elif len(words) == 1 and (words[0].startswith('http') or words[0].startswith('Http')):
		#		pass
		#	else:
			if [words, POS_tags] not in data:
				data.append([words, POS_tags])
			sent = conll_read_sentence(f)

	return data

## Calculate the number of sentences and tokens in a training or test set
def descriptive(data):
	n_sents = len(data)
	n_toks = sum([len(tok[0]) for tok in data])
	return n_sents, n_toks

## Sort original training set based on average token frequency per sentence
## Pick the low frequency sentences
def sort_train(train_data):
	word_freq_dict = {}
	for pair in train_data:
		words = pair[0]
		for w in words:
			if w not in word_freq_dict:
				word_freq_dict[w] = 1
			else:
				word_freq_dict[w] += 1

	train_data_idx_list = [i for i in range(len(train_data))]
	average_word_freq_info = {}
	for idx in train_data_idx_list:
		pair = train_data[idx]
		words = pair[0]
		total_word_freq = sum(word_freq_dict[w] for w in words)
		average_word_freq = total_word_freq / len(words)
		average_word_freq_info[idx] = average_word_freq

	sorted_average_word_freq_info = sorted(average_word_freq_info.items(), key = lambda item: item[1])
	sorted_train_data = [train_data[combo[0]] for combo in sorted_average_word_freq_info]

	return sorted_train_data

## Generate initial training set to start with
## Along with the corresponding selection set
## One initial training sets (for now) for each size 
#def generate_initial_train_select(sorted_train_data, initial_size, treebank):
def generate_initial_train_select(train_data, initial_size, treebank):
	train_set = []
	n_toks = 0 # actual initial training size, measured as the number of tokens
	i = 0
	try:
		while n_toks < initial_size:
		#	pair = sorted_train_data[i]
			pair = train_data[i]
			train_set.append(pair)
			n_toks += len(pair[0])
			i += 1
	except:
		print('Check; Possibly not enough training data')
	#	print(descriptive(sorted_train_data))
		print(descriptive(train_data))
		print('\n')
		return None
		
	if i != 0:
	#	select_set = sorted_train_data[i : ]
		select_set = train_data[i : ]
		if select_set[0] == train_set[-1]:
			print('Check your select and training set for overlap!' + '\n')
			print(treebank, select_set[0], train_set[-1])

	train_set_input_file = io.open('pos_data/' + treebank + '/train.' + str(initial_size) + '.input', 'w')
	train_set_output_file = io.open('pos_data/' + treebank + '/train.' + str(initial_size) + '.output', 'w')
	for pair in train_set:
		train_set_input_file.write(' '.join(w for w in pair[0]) + '\n')
		train_set_output_file.write(' '.join(POS for POS in pair[1]) + '\n')

	select_set_input_file = io.open('pos_data/' + treebank + '/select.' + str(initial_size) + '.input', 'w')
	select_set_output_file = io.open('pos_data/' + treebank + '/select.' + str(initial_size) + '.output', 'w')
	for pair in select_set:
		select_set_input_file.write(' '.join(w for w in pair[0]) + '\n')
		select_set_output_file.write(' '.join(POS for POS in pair[1]) + '\n')

	return n_toks, len(train_set), len(select_set)


# Generate full test data to text files
def generate_test(test_data, treebank):
	test_set_input_file = io.open('pos_data/' + treebank + '/test.full.input', 'w')
	test_set_output_file = io.open('pos_data/' + treebank + '/test.full.output', 'w')
	for tok in test_data:
		test_set_input_file.write(' '.join(w for w in tok[0]) + '\n')
		test_set_output_file.write(' '.join(w for w in tok[1]) + '\n')

if not os.path.exists('pos_data'):
	os.system('mkdir pos_data')

#exception_file = io.open('pos_exceptions.txt', 'w')

initial_size = int(sys.argv[1])
#stats_file = io.open('pos_data/overall_stats.txt', 'w')
#header = ['treebank', 'total_n_toks', 'train_n_toks', 'dev_n_toks', 'test_n_toks']
#stats_file.write('\t'.join(w for w in header) + '\n')

for treebank in os.listdir('pos_data/'): 
	train_file = ''
	dev_file = ''
	test_file = ''
	for file in os.listdir('/blue/liu.ying/unlabeled_pos/ud-treebanks-v2.14/' + treebank + '/'):
		if 'train.conllu' in file:
			train_file = '/blue/liu.ying/unlabeled_pos/ud-treebanks-v2.14/' + treebank + '/' + file
		if 'dev.conllu' in file:
			dev_file = '/blue/liu.ying/unlabeled_pos/ud-treebanks-v2.14/' + treebank + '/' + file
		if 'test.conllu' in file:
			test_file = '/blue/liu.ying/unlabeled_pos/ud-treebanks-v2.14/' + treebank + '/' + file

	if train_file != '' and test_file != '':
#	if treebank == 'UD_English-GUMReddit':
		test_data = collect_data(test_file)
		if test_data == []:
			print(treebank, 'EMPTY')
			print('\n')
		else:
			print(treebank)
			test_n_sents, test_n_toks = descriptive(test_data)
			train_data = collect_data(train_file)
			train_n_sents, train_n_toks = descriptive(train_data)
			dev_data = ''
			dev_n_sents = 0
			dev_n_toks = 0
			try:
				dev_data = collect_data(dev_file)
				dev_n_sents, dev_n_toks = descriptive(dev_data)
			except:
				pass

			total_n_toks = train_n_toks + dev_n_toks + test_n_toks

#			stats_file.write('\t'.join(str(w) for w in [treebank, total_n_toks, train_n_toks, dev_n_toks, test_n_toks]) + '\n')

		### UD Guidelines ###
		# If you have less than 20K words:
		# Option A: Keep everything as test data. Users will have to do 10-fold cross-validation if they want to train on it.
		# Option B: If there are no larger treebanks of this language, keep almost everything as test data but set aside a small sample (20 to 50 sentences) and call it “train”. Consider translating and annotating the 20 examples from the Cairo Cicling Corpus (CCC) and providing them as the sample.
		# If you have between 20K and 30K words, take 10K as test data and the rest as training data.
		# If you have between 30K and 100K words, take 10K as test data, 10K as dev data and the rest as training data.
		# If you have more than 100K words, take 80% as training data, 10% (min 10K words) as dev data and 10% (min 10K words) as test data.
		
#		if test_n_toks >= 10000: # at least 10K tokens in the test set		
			if not os.path.exists('pos_data/' + treebank):	
				os.system('mkdir pos_data/' + treebank)

		#	sorted_train_data = sort_train(train_data) ### only using the training set (better justified?)
			try:
		#		actual_train_n_toks, actual_train_n_sents, actual_select_n_sents = generate_initial_train_select(sorted_train_data, initial_size, treebank)
				actual_train_n_toks, actual_train_n_sents, actual_select_n_sents = generate_initial_train_select(train_data, initial_size, treebank)
			except:
				print(treebank, total_n_toks, train_n_toks, dev_n_toks, test_n_toks)

			generate_test(test_data, treebank)
