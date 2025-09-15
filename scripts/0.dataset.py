# To run,
# python scripts/0.al_dataset.py 500
# 500 is the initial training set size
# While the initial training set size may for 0.al_dataset.py
# This script, 0.random_dataset.py only needs to be run once

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


## Generate training sets via random sampling; the first training set, i.e., select0, is also the initial training set for active learning
## n is the number of random samples to be generated per size
def generate_train(original_train_data, initial_size, treebank, max_size, select_interval, n = 3):
	print(len(original_train_data))
	for i in range(n):
		train_data = original_train_data.copy()  # Make a copy of the original training data
		random.shuffle(train_data)

		sizes = [initial_size]
		size = initial_size
		
		os.makedirs('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/', exist_ok=True)
		os.makedirs('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/' + '/random/', exist_ok=True)
		os.makedirs('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/' + '/random/' + str(select_interval), exist_ok=True)
		os.makedirs('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/' + '/al/', exist_ok=True)
		os.makedirs('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/' + '/al/' + str(select_interval), exist_ok=True)
		os.makedirs('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/random/' + str(select_interval) + '/select0', exist_ok = True)
		os.makedirs('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/al/' + str(select_interval) + '/select0', exist_ok = True)

		while size < max_size:
			size += 250
			sizes.append(size)
	
		if sizes[-1] > max_size:
			sizes = sizes[ : -1]

		train_set =[]
		n_toks = 0
		select_size = 0
		for size in sizes:
			os.makedirs('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/random/' + str(select_interval) + '/select' + str(select_size), exist_ok = True)

			try:
				while n_toks < size - 10:
					pair = random.choices(train_data)[0]
					train_data.remove(pair)
					if pair not in train_set:
						train_set.append(pair)
						n_toks += len(pair[0])
						if len(train_data) == 0:
							break
			except:
				pass

			train_set_input_file = io.open('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/random/' + str(select_interval) + '/select' + str(select_size) + '/train.' + str(initial_size) + '.input', 'w')
			train_set_output_file = io.open('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/random/' + str(select_interval) + '/select' + str(select_size) + '/train.' + str(initial_size) + '.output', 'w')
			for pair in train_set:
				train_set_input_file.write(' '.join(w for w in pair[0]) + '\n')
				train_set_output_file.write(' '.join(POS for POS in pair[1]) + '\n')
			train_set_input_file.close()
			train_set_output_file.close()
			
			if select_size == 0:
				print(len(train_set))
				print('Writing selection file for select0; Number of sentences: ', len(train_data))				
				select_set = train_data  # Define select_set as train_set or appropriate data
				select_set_input_file = io.open('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/al/' + str(select_interval) + '/select0/select.' + str(initial_size) + '.input', 'w')
				select_set_output_file = io.open('pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/al/' + str(select_interval) + '/select0/select.' + str(initial_size) + '.output', 'w')
				for pair in select_set:
					select_set_input_file.write(' '.join(w for w in pair[0]) + '\n')
					select_set_output_file.write(' '.join(POS for POS in pair[1]) + '\n')
				select_set_input_file.close()
				select_set_output_file.close()
					
			select_size += 250
	
	for i in range(n):
		os.system('cp pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/random/' + str(select_interval) + '/select0/train.' + str(initial_size) + '.input' + ' pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/al/' + str(select_interval) + '/select0/')
		os.system('cp pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/random/' + str(select_interval) + '/select0/train.' + str(initial_size) + '.output' + ' pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/al/' + str(select_interval) + '/select0/')

# Generate full test data to text files
def generate_test(test_data, treebank):
	test_set_input_file = io.open('pos_data/' + treebank + '/test.full.input', 'w')
	test_set_output_file = io.open('pos_data/' + treebank + '/test.full.output', 'w')
	for tok in test_data:
		test_set_input_file.write(' '.join(w for w in tok[0]) + '\n')
		test_set_output_file.write(' '.join(w for w in tok[1]) + '\n')

os.makedirs('pos_data', exist_ok=True)
	
initial_size = int(sys.argv[1])
#stats_file = io.open('pos_data/overall_stats.txt', 'w')
#header = ['treebank', 'total_n_toks', 'train_n_toks', 'dev_n_toks', 'test_n_toks']
#stats_file.write('\t'.join(w for w in header) + '\n')

select_interval = 500 #sys.argv[2]

for treebank in os.listdir('/blue/liu.ying/unlabeled_pos/pos_data/'):
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
		### Generating gold-standard test set
		test_data = collect_data(test_file)
		if test_data == []:
			print(treebank, 'EMPTY')
			print('\n')
		else:
			print(treebank)
			os.makedirs('pos_data/' + treebank + '/', exist_ok=True)
			os.makedirs('pos_data/' + treebank + '/' + str(initial_size), exist_ok=True)
	
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

			generate_test(test_data, treebank)
					
			max_size = 0
			if train_n_toks <= 100000:
				max_size = train_n_toks
			else:
				max_size = 100000
		
			generate_train(train_data, initial_size, treebank, max_size, select_interval)

