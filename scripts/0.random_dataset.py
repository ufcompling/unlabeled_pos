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


## Generate training sets via random sampling
## n is the number of random samples to be generated per size
def random_train(train_data, initial_size, treebank, max_size, select_interval, n = 3):
	for i in range(n):
		random.shuffle(train_data)

		sizes = [initial_size]
		size = initial_size
		try:	
			os.system('mkdir pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/random/' + str(select_interval) + '/select0')
		except:
			pass

		while size < max_size:
			size += 250
			sizes.append(size)
	
		if sizes[-1] > max_size:
			sizes = sizes[ : -1]

		train_set =[]
		n_toks = 0
		select_size = 0
		for size in sizes:
			try:
				os.system('mkdir pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/random/' + str(select_interval) + '/select' + str(select_size))
			except:
				pass

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

			select_size += 250

initial_size = int(sys.argv[1])
select_interval = 250 #sys.argv[2]

for treebank in os.listdir('pos_data/'):
	train_toks = 0
	max_size = 0
	for file in os.listdir('ud-treebanks-v2.14/' + treebank + '/'):
		if file.endswith('train.txt'):
			with open('ud-treebanks-v2.14/' + treebank + '/' + file) as full_f:
				for line in full_f:
					toks = line.strip().split()
					train_toks += len(toks)

	if train_toks <= 100000:
		max_size = train_toks
	else:
		max_size = 100000

	train_file = ''

	for file in os.listdir('ud-treebanks-v2.14/' + treebank + '/'):
		if 'train.conllu' in file:
			train_file = 'ud-treebanks-v2.14/' + treebank + '/' + file

	if train_file != '': # and treebank in ['UD_Turkish-Atis']: #not in ['UD_Afrikaans-AfriBooms', 'UD_Hebrew-HTB', 'UD_Italian-ParTUT', 'UD_Faroese-FarPaHC', 'UD_Armenian-ArmTDP', 'UD_Basque-BDT', 'UD_Slovenian-SSJ', 'UD_English-EWT', 'UD_Persian-PerDT', 'UD_Slovenian-SST', 'UD_Turkish-Tourism']:
		print(treebank)
		train_data = collect_data(train_file)
		
		try:
			os.system('mkdir pos_data/' + treebank + '/' + str(initial_size))
			os.system('mkdir pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/')
			os.system('mkdir pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/' + '/random/')
			os.system('mkdir pos_data/' + treebank + '/' + str(initial_size) + '/' + str(i) + '/' + '/random/' + str(select_interval))
		except:
			pass
		random_train(train_data, initial_size, treebank, max_size, select_interval)
