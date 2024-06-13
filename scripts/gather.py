# e.g., python3 scripts/gather.py al_trainselect/ surSeg

import io, os, sys

datadir = '/blue/liu.ying/unlabeled_pos/pos_data/'

select_interval = '250' #sys.argv[1]

sizes = ['500']

task = 'pos'
outfile = open(task + '_results.txt', 'w')

max_size = 20000

header = ['Treebank', 'Size', 'Select_interval', 'Select_size', 'Model', 'Metric', 'Value']
outfile.write(' '.join(tok for tok in header) + '\n')

for size in sizes:
	for treebank in os.listdir(datadir):
		if size in os.listdir(datadir + treebank):
			for arch in ['transformer']: #, 'transformer_tiny', 'lstm']:
				for method in ['al', 'tokenfreq']:
					iterations = int(max_size / int(select_interval))
					for i in range(iterations):
						select = str(i * int(select_interval))
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

							info = [lg, task, size, select_interval, select, arch, 'precision', precision]
							outfile.write(' '.join(str(tok) for tok in info) + '\n')
							info = [lg, task, size, select_interval, select, arch, 'recall', recall]
							outfile.write(' '.join(str(tok) for tok in info) + '\n')
							info = [lg, task, size, select_interval, select, arch, 'f1', f1]
							outfile.write(' '.join(str(tok) for tok in info) + '\n')

						select += int(select_interval)

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

	info = [lg, task, size, select_interval, select, arch, 'precision', precision]
	outfile.write(' '.join(str(tok) for tok in info) + '\n')
	info = [lg, task, size, select_interval, select, arch, 'recall', recall]
	outfile.write(' '.join(str(tok) for tok in info) + '\n')
	info = [lg, task, size, select_interval, select, arch, 'f1', f1]
	outfile.write(' '.join(str(tok) for tok in info) + '\n')

