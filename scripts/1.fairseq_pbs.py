import io, os, sys

second_string = '''#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liu.ying@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                    
#SBATCH --mem=8gb                     # Job memory request
#SBATCH --time=24:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log

pwd; hostname; date

module load conda
mamba activate fairseq-env
module load python3

cd /blue/liu.ying/unlabeled_pos/

'''

select_interval = '250' #sys.argv[1]

if not os.path.exists('pbs/'):
	os.system('mkdir pbs/')

#sizes = ['500', '1000', '1500', '2000']
sizes = ['50']#, '100', '300', '500', '1000']

task = 'pos'
#max_size = 100000

#treebanks_select = [sys.argv[1]]

#for treebank in treebanks_select:
for treebank in os.listdir('/blue/liu.ying/unlabeled_pos/pos_data'):
	if treebank.startswith('UD') is True:
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

		for size in sizes:
			for arch in ['crf']: #'transformer']: #, 'transformer_tiny', 'lstm']:
				for method in ['al']:#, 'tokenfreq']:
					iterations = int(max_size / int(select_interval))
					with open('pbs/' + treebank + size + '_' + str(max_size) + '_' + arch + '_' + method + '.pbs', 'w') as f:
						first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + treebank + '_' + size + '_' + str(max_size) + '_' + arch + '    # Job name'

						f.write(first_string + '\n')
						f.write(second_string + '\n')

						for i in range(iterations):
							select = str(i * int(select_interval))
							if int(size) + int(select) <= max_size:# and int(size) + int(select) > 20000: # and int(select) >= 10000:
						
								f.write('python scripts/' + arch + '.py /blue/liu.ying/unlabeled_pos/' + task + '_data/ ' + treebank + ' ' + size + ' ' + select_interval + ' ' + select + ' ' + arch + ' ' + method + '\n')
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
			if size == '50':
				select = 'all'
				for arch in ['crf']: #, 'transformer_tiny', 'lstm']:
					with open('pbs/' + treebank + '_select' + select + '_' + arch +'.pbs', 'w') as f:
						first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + treebank + '_' + size + '_' + select + '_' + arch + '    # Job name'
						f.write(first_string + '\n')
						f.write(second_string + '\n')

						f.write('python scripts/' + arch + '.py /blue/liu.ying/unlabeled_pos/' + task + '_data/ ' + treebank + ' ' + size + ' ' + select_interval + ' ' + select + ' ' + arch + ' al' + '\n')
					#	f.write('\n')
			
					#	f.write('module load python3' + '\n')
					#	f.write('\n')
			#	f.write('python scripts/eval.py /blue/liu.ying/unlabeled_pos/' + task + '_data/ ' + treebank + ' ' + size + ' ' + select_interval + ' ' + select + ' ' + arch + ' al' + '\n')
					#	f.write('\n')
			
						f.write('date' + '\n')
						f.write('\n')

'''
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1
'''