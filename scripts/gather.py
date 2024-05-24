# e.g., python3 scripts/gather.py al_trainselect/ surSeg

import io, os, sys

datadir = sys.argv[1]

arch = 'transformer'

lgs = ['btz', 'cho', 'lez', 'ntu', 'tau', 'bdg']
sizes = ['500', '1000', '1500', '2000']

header = ['Language', 'Task', 'Size', 'Select_size', 'Select_interval', 'Model', 'Metric', 'Value']

task = sys.argv[2]

for select_interval in ['25', '100', '500']:
	outfile = open(task + '_results.txt', 'w')
	outfile.write(' '.join(tok for tok in header) + '\n')
	for lg in lgs:
		for size in sizes:
			select = 0
			while select <= 2000:
				evaluation_file = datadir + lg + '_' + task + size + '/' + select_interval + '/select' + str(select) + '/' + arch + '/eval.txt'
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

				else:
					if select + int(size) <= 2500:
						print(lg, task, size, select_interval, select)

				select += int(select_interval)

'''
			select = 'all'
			evaluation_file = datadir + lg + '_' + task + size + '/' + select_interval + '/select' + str(select) + '/' + arch + '/eval.txt'
			print(evaluation_file)
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

			else:
				print(lg, task, size, select_interval, select)
'''