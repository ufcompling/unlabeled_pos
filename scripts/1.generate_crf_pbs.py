# To run,
# python scripts/1.generate_crf_pbs.py 500

import io, os, sys

second_string = '''#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liu.ying@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                    
#SBATCH --mem=7gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --account=ufdatastudios
#SBATCH --qos=ufdatastudios
pwd; hostname; date

module load conda
mamba activate fairseq-env

cd /orange/ufdatastudios/zoey.liu/unlabeled_pos/

'''

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


select_interval = '500'

if not os.path.exists('pbs/'):
	os.system('mkdir pbs/')

sizes = [sys.argv[1]]

task = 'pos'

for treebank in os.listdir('pos_data/'):
	if treebank.startswith('UD') is True:
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
		print(treebank, max_size)
		for size in sizes:		
			for arch in ['crf']: #'transformer']: #, 'transformer_tiny', 'lstm']:
				for method in ['al', 'random']:#, 'tokenfreq']:
					iterations = int(max_size / int(select_interval))
					for z in range(3):
						with open('pbs/' + treebank + size + '_' + str(max_size) + '_' + arch + '_' + method + '_' + str(z) + '.pbs', 'w') as f:
							first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + treebank + size + '_' + str(max_size) + '_' + arch + '_' + method + '_' + str(z) + '    # Job name'

							f.write(first_string + '\n')
							f.write(second_string + '\n')
						
							for i in range(iterations):
								select = str(i * int(select_interval))
								if int(size) + int(select) <= max_size:# and int(size) + int(select) > 20000: # and int(select) >= 10000:
						
									f.write('python scripts/' + arch + '_' + method + '.py /orange/ufdatastudios/zoey.liu/unlabeled_pos/' + task + '_data/ ' + treebank + ' ' + size + ' ' + select_interval + ' ' + select + ' ' + arch + ' ' + method + ' ' + str(z) + '\n')
						#		f.write('\n')
			
						#		f.write('module load python3' + '\n')
						#		f.write('\n')
						#		f.write('python scripts/eval.py /blue/liu.ying/unlabeled_pos/' + task + '_data/ ' + treebank + ' ' + size + ' ' + select_interval + ' ' + select + ' ' + arch + ' ' + method + '\n')
						#		f.write('\n')
						#		f.write('module load conda' + '\n')
						#		f.write('mamba activate fairseq-env' + '\n')
									f.write('\n')
			
							f.write('date' + '\n')
							f.write('\n')


			# Only need to run one of these pbs files since the total amount of data available is the same
#			if size == '500':
#				select = 'all'
#				for arch in ['crf']: #, 'transformer_tiny', 'lstm']:
#					with open('pbs/' + treebank + '_select' + select + '_' + arch +'.pbs', 'w') as f:
#						first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + treebank + '_' + size + '_' + select + '_' + arch + '    # Job name'
#						f.write(first_string + '\n')
#						f.write(second_string + '\n')

#						f.write('python scripts/' + arch + '.py /blue/liu.ying/unlabeled_pos/' + task + '_data/ ' + treebank + ' ' + size + ' ' + select_interval + ' ' + select + ' ' + arch + ' al' + '\n')

#						f.write('date' + '\n')
#						f.write('\n')


'''
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
'''
