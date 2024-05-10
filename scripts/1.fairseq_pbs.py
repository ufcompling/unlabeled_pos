import io, os

second_string = '''#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=liu.ying@ufl.edu     # Where to send mail	
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                    
#SBATCH --mem=8gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=serial_test_%j.log   # Standard output and error log
#SBATCH --partition=gpu
#SBATCH --gpus=a100:1

pwd; hostname; date

module load conda
mamba activate data_partition
module load fairseq

cd /blue/liu.ying/al_sensitivity/

'''

if not os.path.exists('pbs/'):
	os.system('mkdir pbs/')

sizes = ['50', '100', '500', '1000', '1500', '2000']

for treebank in os.listdir('/blue/liu.ying/al_sensitivity/pos_data'):
	if treebank.startswith('UD') is True:
		for task in ['pos']:
			for method in ['al']:
				for size in sizes:
					for idx in range(10):
						idx += 1
						idx = str(idx)
						select_file = '/blue/liu.ying/al_sensitivity/' + task + '_data/' + method + '/' + treebank + '/select.' + size + '_' + idx '.input'
						max_size = 0
						with open(select_file) as f:
							for line in f:
								max_size += 1

						iterations = int(max_size / 25)
						for i in range(iterations):
							select = str(i * 25)
							for arch in ['transformer']: #, 'transformer_tiny', 'lstm']:
								with open('pbs/' + treebank + size + '_' + idx + '_select' + select + '_' + arch +'.pbs', 'w') as f:
									first_string = '''#!/bin/bash\n#SBATCH --job-name=''' + treebank + '_' + size + '_' + idx + '_' + select + '_' + arch + '    # Job name'

									f.write(first_string + '\n')
									f.write(second_string + '\n')

									f.write('python scripts/fairseq.py /blue/liu.ying/al_sensitivity/' + task + '_data/' + method + '/ ' treebank + ' ' + size + ' ' + idx + ' ' + select + ' ' + arch + '\n')
									f.write('\n')
			
									f.write('date' + '\n')
									f.write('\n')
