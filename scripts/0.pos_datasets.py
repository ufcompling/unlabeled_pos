import io, os, sys
import random

# Read CoNLL-U format
def conll_read_sentence(file_handle):

	sent = []

	for line in file_handle:
		line = line.strip('\n')
		if line.startswith('#') is False:
			toks = line.split("\t")
			if len(toks) != 10 and sent not in [[], ['']]:
				return sent 
			if len(toks) == 10 and '-' not in toks[0] and '.' not in toks[0]:
				if toks[0] == 'q1':
					toks[0] = '1'
				sent.append(toks)

	return None

## Collect data from the training or the test set
def collect_data(file_handle):
	data = []
	with open(file) as f:
		sent = conll_read_sentence(f)
		while sent is not None:
			words = [tok[1] for tok in sent]
			POS_tags = [tok[3] for tok in sent]
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
	sorted_train_data = [train_data[idx] for idx in sorted_average_word_freq_info]

	return sorted_train_data

## Generate initial training set to start with
## Along with the corresponding selection set
## One initial training sets (for now) for each size 
def generate_initial_train_select(sorted_train_data, initial_size, treebank):
	train_set = []
	n_toks = 0 # actual initial training size, measured as the number of tokens
	i = 0
	try:
		while n_toks <= initial_size:
			pair = sorted_train_data[i]
			train_set.append(pair)
			n_toks += len(pair[0])
			i += 1
	except:
		print('Check; Possibly not enough training data')
		print(descriptive(sorted_train_data))
		print('\n')
		
	if i != 0:
		select_set = sorted_train_data[i : ]
		if select_set[0] == train_set[-1]:
			print('Check your select and training set for overlap!' + '\n')

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

try:
	os.system('mkdir pos_data')
except:
	pass

#exception_file = io.open('pos_exceptions.txt', 'w')

initial_size = int(sys.argv[1])
stats_file = io.open('pos_data/overall_stats.txt', 'w')
header = ['treebank', 'train_n_sents', 'train_n_toks', 'dev_n_sents', 'dev_n_toks', 'test_n_sents', 'test_n_toks']

for treebank in os.listdir('ud-treebanks-v2.13/'):
	train_file = ''
	dev_file = ''
	test_file = ''
	for file in os.listdir('ud-treebanks-v2.13/' + treebank + '/'):
		if 'train.conllu' in file:
			train_file = 'ud-treebanks-v2.13/' + treebank + '/' + file
		if 'dev.conllu' in file:
			dev_file = 'ud-treebanks-v2.13/' + treebank + '/' + file
		if 'test.conllu' in file:
			test_file = 'ud-treebanks-v2.13/' + treebank + '/' + file

	if train_file != '' and test_file != '':

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
		test_data = collect_data(test_file)
		test_n_sents, test_n_toks = descriptive(test_data)

		### UD Guidelines ###
		# If you have less than 20K words:
		# Option A: Keep everything as test data. Users will have to do 10-fold cross-validation if they want to train on it.
		# Option B: If there are no larger treebanks of this language, keep almost everything as test data but set aside a small sample (20 to 50 sentences) and call it “train”. Consider translating and annotating the 20 examples from the Cairo Cicling Corpus (CCC) and providing them as the sample.
		# If you have between 20K and 30K words, take 10K as test data and the rest as training data.
		# If you have between 30K and 100K words, take 10K as test data, 10K as dev data and the rest as training data.
		# If you have more than 100K words, take 80% as training data, 10% (min 10K words) as dev data and 10% (min 10K words) as test data.
		
		total_n_toks = train_n_toks + dev_n_toks + test_n_toks
		if total_n_toks <= 30000: # 30K tokens
			print(treebank)
			os.system('mkdir pos_data/' + treebank)

			sorted_train_data = sort_train(train_data)
			actual_train_n_toks, actual_train_n_sents, actual_select_n_sents = generate_initial_train_select(sorted_train_data, initial_size, treebank):
			print(actual_train_n_toks)
			
			generate_test(test_data, treebank)

		print('')

