# e.g., python3 scripts/gather.py al_trainselect/ surSeg

import io, os, sys

datadir = '/blue/liu.ying/unlabeled_pos/pos_data/'

select_interval = '250' #sys.argv[1]

sizes = ['50']#, '1000']

task = 'pos'
outfile = open(size + '_' + task + '_results.txt', 'w')

header = ['Treebank', 'Task', 'Size', 'Select_interval', 'Select_size', 'Total', 'Model', 'Metric', 'Value']
outfile.write(' '.join(tok for tok in header) + '\n')

#treebanks = ['UD_Irish-IDT']

for size in sizes:
	for treebank in os.listdir('ud-treebanks-v2.14'):
		if os.path.exists(datadir + treebank):
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

			if size in os.listdir(datadir + treebank):
				for arch in ['crf']: #, 'transformer_tiny', 'lstm']:
					for method in ['al']:#, 'tokenfreq']:
						iterations = int(max_size / int(select_interval))
						for i in range(iterations):
							select = str(i * int(select_interval))
							total = 0
							train_file = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + str(select) + '/train.' + size + '.input'
							with open(train_file) as f:
								for train_line in f:
									total += len(train_line.strip().split())

							evaluation_file = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + str(select) + '/' + arch + '/eval.txt'
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

								info = [treebank, task, size, select_interval, select, total, arch, 'precision', precision]
								outfile.write(' '.join(str(tok) for tok in info) + '\n')
								info = [treebank, task, size, select_interval, select, total, arch, 'recall', recall]
								outfile.write(' '.join(str(tok) for tok in info) + '\n')
								info = [treebank, task, size, select_interval, select, total, arch, 'f1', f1]
								outfile.write(' '.join(str(tok) for tok in info) + '\n')

				#		select += int(select_interval)

							else:
								select_file = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + str(select) + '/select.' + size + '.input'
								if not os.path.exists(select_file):
									print(select_file)			
									previous_select_file = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + str(int(select) - 250) + '/select.' + size + '.input'
									if os.path.exists(previous_select_file):
										select_total = 0
										with open(previous_select_file) as f:
											for line in f:
												toks = line.strip().split()
												select_total += len(toks)
										if not (total >= max_size or select_total < 250):
											print(evaluation_file)
								else:
									if select <= 500:
										print(evaluation_file)
								

select = 'all'
evaluation_file = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + str(select) + '/' + arch + '/eval.txt'

if os.path.exists(evaluation_file):
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

	info = [treebank, task, size, select_interval, select, arch, 'precision', precision]
	outfile.write(' '.join(str(tok) for tok in info) + '\n')
	info = [treebank, task, size, select_interval, select, arch, 'recall', recall]
	outfile.write(' '.join(str(tok) for tok in info) + '\n')
	info = [treebank, task, size, select_interval, select, arch, 'f1', f1]
	outfile.write(' '.join(str(tok) for tok in info) + '\n')

