import io, os, sys

second_string = '''#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liu.ying@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                    
#SBATCH --mem=8gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log

pwd; hostname; date

module load python3

cd /blue/liu.ying/unlabeled_pos/

'''

if not os.path.exists('pbs/'):
	os.system('mkdir pbs/')


task = 'pos'
max_size = 100000

treebanks_select = [sys.argv[1]]
sizes = [sys.argv[2]]
select_interval = sys.argv[3]

for treebank in treebanks_select:
#for treebank in os.listdir('/blue/liu.ying/unlabeled_pos/pos_data'):
	if treebank.startswith('UD') is True:
		for size in sizes:
			for arch in ['crf']: #, 'transformer_tiny', 'lstm']:
				for method in ['al']:#, 'tokenfreq']:
					iterations = int(max_size / int(select_interval))
					with open('pbs/' + treebank + size + '_' + str(max_size) + '_' + arch + '_' + method + '.pbs', 'w') as f:
						first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + treebank + '_' + size + '_' + str(max_size) + '_' + arch + '    # Job name'

						f.write(first_string + '\n')
						f.write(second_string + '\n')

						for i in range(iterations):
							select = str(i * int(select_interval))
							if int(size) + int(select) <= max_size and int(size) + int(select) > 20000: # and int(select) >= 10000:
						
								f.write('python scripts/crf.py /blue/liu.ying/unlabeled_pos/' + task + '_data/ ' + treebank + ' ' + size + ' ' + select_interval + ' ' + select + ' ' + arch + ' ' + method + '\n')
								f.write('\n')
			
						f.write('date' + '\n')
						f.write('\n')

		# Only need to run one of these pbs files since the total amount of data available is the same
		select = 'all'
		for arch in ['crf']: #, 'transformer_tiny', 'lstm']:
			with open('pbs/' + treebank + size + '_select' + select + '_' + arch +'.pbs', 'w') as f:
				first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + treebank + '_' + size + '_' + select + '_' + arch + '    # Job name'

				f.write(first_string + '\n')
				f.write(second_string + '\n')

				f.write('python scripts/crf.py /blue/liu.ying/unlabeled_pos/' + task + '_data/ ' + treebank + ' ' + size + ' ' + select_interval + ' ' + select + ' ' + arch + ' al' + '\n')
				f.write('\n')
			
				f.write('date' + '\n')
				f.write('\n')

'''
### Combine pbs file for each language, given a start size, a select interval, and a select size

iterations = int(max_size / int(select_interval))
for treebank in treebanks_select:
	for method in ['al', 'tokenfreq']:
		for size in sizes:
			together_file = open('pbs/' + treebank + '_' + size + '_' + select_interval +  '_' + method + '.sh', 'w') # doing sbatch all together
			c = 0
			for i in range(iterations):
				select = str(i * int(select_interval))
				for arch in ['transformer']:
					evaluation_file = '/blue/liu.ying/unlabeled_pos/' + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + arch + '/eval.txt'
					if os.path.exists(evaluation_file) and os.stat(evaluation_file).st_size != 0:
						pass
					else:
						together_file.write('sbatch pbs/' + treebank + size + '_select' + select + '_' + arch + '_' + method + '.pbs' + '\n')
						c += 1

			if c == 0: ## deleting empty together_file
				os.system('rm ' + 'pbs/' + treebank + '_' + size + '_' + select_interval +  '_' + method + '.sh')

'''

'''
## For each start size, combine pbs files for all languages for a given select interval and select size
iterations = int(max_size / int(select_interval))
for i in range(iterations):
	select = str(i * int(select_interval))
	for size in sizes:
		together_file = open('pbs_new/' + task + size + '_' + select_interval + '_select' + select + '.sh', 'w') # doing sbatch all together
		c = 0
		for lg in lgs:			
			for arch in ['transformer']:
				if lg + '_' + task + size + '_' + select_interval + '_select' + select + '_' + arch +'.pbs' in os.listdir('pbs_new/'):
					evaluation_file = '/blue/liu.ying/al_morphseg/al_trainselect/' + lg + '_' + task + size + '/' + select_interval + '/select' + str(select) + '/' + arch + '/eval.txt'
					if os.path.exists(evaluation_file) and os.stat(evaluation_file).st_size != 0:
						pass
					else:
						together_file.write('sbatch pbs_new/' + lg + '_' + task + size + '_' + select_interval + '_select' + select + '_' + arch +'.pbs' + '\n')
						c += 1
		if c == 0:
			os.system('rm ' + 'pbs_new/' + task + size + '_' + select_interval + '_select' + select + '.sh')
'''