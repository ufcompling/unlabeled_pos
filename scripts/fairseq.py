#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import sys, statistics
from datetime import datetime
import torch 

def main(datadir, treebank, size, select_interval, select, arch, method):
	# Determine the assigned GPU device dynamically
    device_id = torch.cuda.current_device()  # Get the current GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)  # Set the environment variable dynamically

	subprocess.run(['mkdir', '-p', datadir + treebank])
	subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size])
	subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size + '/' + method])
	subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size + '/' + method + '/' + select_interval])
	subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select])
	subprocess.run(['mkdir', '-p', datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/' + arch])

	sub_datadir = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + select + '/'
	previous_datadir = ''
	if select not in ['0', 'all']:
		previous_datadir = datadir + treebank + '/' + size + '/' + method + '/' + select_interval + '/select' + str(int(select) - int(select_interval)) + '/'

	if select == '0':
		os.system('cp ' + datadir + treebank + '/train.' + size + '.input ' + sub_datadir)
		os.system('cp ' + datadir + treebank + '/train.' + size + '.output ' + sub_datadir)
		os.system('cp ' + datadir + treebank + '/select.' + size + '.input ' + sub_datadir)
		os.system('cp ' + datadir + treebank + '/select.' + size + '.output ' + sub_datadir)

	elif select == 'all':
		os.system('cat ' + datadir + treebank + '/train.' + size + '.input ' + datadir + treebank + '/select.' + size + '.input >' + sub_datadir + 'train.' + size + '.input')
		os.system('cat ' + datadir + treebank + '/train.' + size + '.input ' + datadir + treebank + '/select.' + size + '.output >' + sub_datadir + 'train.' + size + '.output')

	else:
		os.system('cat ' + previous_datadir + 'train.' + size + '.input ' + previous_datadir + '/increment.input >' + sub_datadir + 'train.' + size + '.input')
		os.system('cat ' + previous_datadir + 'train.' + size + '.output ' + previous_datadir + '/increment.output >' + sub_datadir + 'train.' + size + '.output')
		os.system('mv ' + previous_datadir + 'residual.input ' + sub_datadir + 'select.' + size + '.input')
		os.system('mv ' + previous_datadir + 'residual.output ' + sub_datadir + 'select.' + size + '.output')

#	lang = treebank.split('-')[0].split('_')[1]
#	SRC = lang + '_' + size + '.input'
#	TGT = lang + '_' + size + '.output'
	SRC = size + '.input'
	TGT = size + '.output'
	TESTIN = datadir + treebank + '/test.full.input'
	TO_PREDICT = sub_datadir + 'select.'+ size + '.input'

	### Collecting the pool of words to select from
	select_input = []
	select_output = []
	select_combo = []
	with open(sub_datadir + '/select.'+ size + '.input') as f:
		for line in f:
			toks = line.strip()
			select_input.append(toks)

	
	with open(sub_datadir + '/select.'+ size + '.output') as f:
		for line in f:
			toks = line.strip()
			select_output.append(toks)

	select_combo = [select_input[i] + '_' + select_output[i] for i in range(len(select_input))]

	confidence_dict = {}

	for seed in ['1', '2', '3']:
		PREDDIR = sub_datadir + arch + '/' + seed + '/preds/' ## seed
		FROMDIR = sub_datadir + arch + '/' + seed + '/data-bin/'
		SAVEDIR = sub_datadir + arch + '/' + seed + '/checkpoints/'
	#    SCOREDIR = datadir + experiment + '/' + task + '/' + arch + '/' + seed + '/scores/'
	
		MODELPATH = SAVEDIR + 'checkpoint_best.pt'   
	
		GUESSPATH = PREDDIR + treebank + '_' + size + '.testpredict'
		U_CONFIDENCE = ''
		U_PREDICTIONS = ''
		if method == 'al' and select != 'all':
			U_CONFIDENCE = PREDDIR + treebank + '_' + size + '.confidence'
			U_PREDICTIONS = PREDDIR + treebank + '_' + size + '.predict'
		else:
			pass
	
		# create folders
		subprocess.run(['mkdir', '-p', PREDDIR])
		subprocess.run(['mkdir', '-p', FROMDIR])
		subprocess.run(['mkdir', '-p', SAVEDIR])
	#    subprocess.run(['mkdir', '-p', SCOREDIR])
	
		# clear checkpoint and model folders from previous runs
		subprocess.run(['rm', '-rf', SAVEDIR])
		subprocess.run(['rm', '-rf', FROMDIR])

		# preprocess
		subprocess.call(['fairseq-preprocess',
						'--source-lang=' + SRC,
						'--target-lang=' + TGT,
						'--trainpref='+ sub_datadir + '/train',
						'--validpref='+ sub_datadir + '/train',
						#'--testpref='+TEXTDIR+'test',
						'--destdir=' + FROMDIR])
	
		print("##################### PREPROCESSED", size, "#####################")
	
		# train
		#Transformer Parameters from Wu 2021 et al.: ACTIVATION_DROPOUT=0.3, ACTIVATION_FN='RELU', ADAM_BETAS='(0.9, 0.98)', ADAM_EPS=1E-08, ADAPTIVE_INPUT=FALSE, ADAPTIVE_SOFTMAX_CUTOFF=NONE, ADAPTIVE_SOFTMAX_DROPOUT=0, CROSS_SELF_ATTENTION=FALSE, DECODER_ATTENTION_HEADS=4, DECODER_EMBED_DIM=256, DECODER_EMBED_PATH=NONE, DECODER_FFN_EMBED_DIM=1024, DECODER_INPUT_DIM=256, DECODER_LAYERDROP=0, DECODER_LAYERS=4, DECODER_LEARNED_POS=FALSE, DECODER_NORMALIZE_BEFORE=TRUE, DECODER_OUTPUT_DIM=256, DISTRIBUTED_WRAPPER='DDP', DROPOUT=0.3, ENCODER_ATTENTION_HEADS=4, ENCODER_EMBED_DIM=256, ENCODER_FFN_EMBED_DIM=1024, ENCODER_LAYERDROP=0, ENCODER_LAYERS=4, ENCODER_LEARNED_POS=FALSE, ENCODER_NORMALIZE_BEFORE=TRUE, EVAL_BLEU=FALSE, EVAL_BLEU_ARGS=NONE, EVAL_BLEU_DETOK='SPACE', EVAL_BLEU_DETOK_ARGS=NONE, EVAL_BLEU_PRINT_SAMPLES=FALSE, EVAL_BLEU_REMOVE_BPE=NONE, EVAL_TOKENIZED_BLEU=FALSE, FIXED_VALIDATION_SEED=NONE, IGNORE_PREFIX_SIZE=0, LABEL_SMOOTHING=0.1, LAYERNORM_EMBEDDING=FALSE, LEFT_PAD_SOURCE='TRUE', LEFT_PAD_TARGET='FALSE', LOAD_ALIGNMENTS=FALSE, MAX_SOURCE_POSITIONS=1024, MAX_TARGET_POSITIONS=1024, NO_CROSS_ATTENTION=FALSE, NO_SCALE_EMBEDDING=FALSE, NO_TOKEN_POSITIONAL_EMBEDDINGS=FALSE, NUM_BATCH_BUCKETS=0, 
		# pipeline_balance=None, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_encoder_balance=None, 
		#QUANT_NOISE_PQ=0, QUANT_NOISE_PQ_BLOCK_SIZE=8, QUANT_NOISE_SCALAR=0, REPORT_ACCURACY=FALSE, SHARE_ALL_EMBEDDINGS=FALSE, SHARE_DECODER_INPUT_OUTPUT_EMBED=TRUE, TIE_ADAPTIVE_WEIGHTS=FALSE, TRUNCATE_SOURCE=FALSE, UPSAMPLE_PRIMARY=1, USE_OLD_ADAM=FALSE=, WARMUP_INIT_LR=-1, WARMUP_UPDATES=1000, WEIGHT_DECAY=0.0)

		#subprocess.call(['./train.sh', dataset])
		if not os.path.exists(MODELPATH):
			subprocess.call(['fairseq-train', 
					 FROMDIR,
					'--save-dir=' + SAVEDIR,
					#'--restore-file=' + MODELPATH
					'--source-lang=' + SRC,
					'--target-lang=' + TGT,
					'--arch=' + arch,
					#'--pipeline-encoder-balance=4',
					#'--pipeline-decoder-balance=4',
					'--batch-size=400',
					'--batch-size-valid=400',
					'--clip-norm=1.0',
					"--criterion=label_smoothed_cross_entropy",
					#"--ddp-backend=c10d",
					"--ddp-backend=legacy_ddp",
					'--lr=[0.001]',
					"--lr-scheduler=inverse_sqrt",
					'--max-update=6000',
					'--optimizer=adam',
					'--seed=' + seed,
					'--save-interval=10',
					'--patience=10', # early stopping
					'--skip-invalid-size-inputs-valid-test',
					'--no-epoch-checkpoints',
					'--no-last-checkpoints',
					'--device-id', str(device_id)  # use --device-id to make sure all tensors are on the same device.
					]) 
	
		print("##################### TRAINED", size, "#####################")
	
		# test
		testing = subprocess.Popen(['fairseq-interactive',
							  FROMDIR,
							  '--path', MODELPATH,
							  '--source-lang='+SRC,
							  '--target-lang='+TGT,
							  '--skip-invalid-size-inputs-valid-test',
							  '--match-source-len',
							#  '--cpu'
							  '--device-id', str(device_id)
							 ],
						#	 shell=True,
							 stdin=open(TESTIN),
							 stdout=subprocess.PIPE)

		# Write test predictions to file
		with open(GUESSPATH, 'w')  as predout:
			writeline = []
			testout = testing.stdout.readlines()
			for line in testout:
				line = line.decode('utf-8')
				if line[0] == 'H':
					writeline.append(line.split('\t')[2].strip())
			predout.write('\n'.join(writeline))
		
		print("##################### TESTED", size, "#####################")
		
		# predict on unlabeled data and write predictions to file
		if method == 'al' and select != 'all':
			predicting = subprocess.Popen(['fairseq-interactive',
							  FROMDIR,
							  '--path', MODELPATH,
							  '--source-lang='+SRC,
							  '--target-lang='+TGT,
							  '--skip-invalid-size-inputs-valid-test',
							  '--match-source-len',
							  '--device-id', str(device_id)
						#	  '--cpu'
							 ],
						#	 shell=True,
							 stdin=open(TO_PREDICT),
							 stdout=subprocess.PIPE)


			with open(U_CONFIDENCE, 'w') as predwconfidence, open(U_PREDICTIONS, 'w') as predict:
				writeline = []
				all_writeline = []
				predictions = predicting.stdout.readlines()
				predictions = [str(line.decode('utf-8').strip()) for line in predictions]
				for lineU in predictions:
					if lineU[0] == 'H':
						writeline.append(lineU.split('\t')[2].strip()) # no confidence
						all_writeline.append(str(lineU.strip())) # w confidence
				predict.write('\n'.join(writeline)) # fixed this from the original code
				predwconfidence.write('\n'.join(all_writeline))
		

			print("##################### PREDICTED", size, "#####################")

			## Extracting confidence scores
			confidence_scores = []
			with open(U_CONFIDENCE) as f:
				for line in f:
					toks = line.strip().split('\t')
					confidence_scores.append(toks[1])
		
			for i in range(len(select_output)):
				pair = select_input[i] + '_' + select_output[i]
				confidence_score = confidence_scores[i]
				if seed == '1':
					confidence_dict[pair] = [float(confidence_score)]
				else:
					confidence_dict[pair].append(float(confidence_score))

		else:
			pass


		### Cleaning space
	#	subprocess.run(['rm', '-rf', FROMDIR])

	### Selecting words for active learning method

	if method == 'al' and select != 'all':
		for k, v in confidence_dict.items():
			confidence_dict[k] = statistics.mean(v)
	
		sorted_confidence_dict = sorted(confidence_dict.items(), key = lambda item: item[1])
		increment_sents = sorted_confidence_dict[ : int(select_interval)]
		n_toks = 0
		i = 0
		while n_toks <= int(select_interval):
			combo = sorted_confidence_dict[i]
			increment_sents.append(combo)
			pair = combo[0].split('_')
			all_toks = pair[0].split()
			n_toks += len(all_toks)
			i += 1
		print('Tokens: ' + str(n_toks))
		print('\n')
		print('\n')

		## Checking confidence score output
		for combo in increment_sents:
			print(combo[0], confidence_dict[combo[0]])
		print('')
		print('')
		for combo in sorted_confidence_dict[i : i + 5]:
			print(combo[0], confidence_dict[combo[0]])

		increment_input = open(sub_datadir + 'increment.input', 'w')
		increment_output = open(sub_datadir + 'increment.output', 'w')
		increment_pairs = []
		for combo in increment_sents:
			pair = combo[0]
			increment_pairs.append(pair)
			pair = pair.split('_')
			increment_input.write(pair[0] + '\n')
			increment_output.write(pair[1] + '\n')

		print('')
		print('Start writing residual output' + '\n')
		residual_input = open(sub_datadir + 'residual.input', 'w')
		residual_output = open(sub_datadir + 'residual.output', 'w')
		for pair in select_combo:
			if pair not in increment_pairs:
				pair = pair.split('_')
				residual_input.write(pair[0] + '\n')
				residual_output.write(pair[1] + '\n')

	if method == 'tokenfreq':
		increment_input = open(sub_datadir + 'increment.input', 'w')
		increment_output = open(sub_datadir + 'increment.output', 'w')

		n_toks = 0
		i = 0
		while n_toks <= int(select_interval):
			combo = select_combo[i]
			pair = combo.split('_')
			all_toks = pair[0].split()
			n_toks += len(all_toks)
			increment_input.write(pair[0] + '\n')
			increment_output.write(pair[1] + '\n')
			i += 1 

		residual_input = open(sub_datadir + 'residual.input', 'w')
		residual_output = open(sub_datadir + 'residual.output', 'w')
		residual_start_idx = i
		for i in range(len(select_combo[residual_start_idx : ])):
			combo = select_combo[i]
			pair = combo.split('_')
			residual_input.write(pair[0] + '\n')
			residual_output.write(pair[1] + '\n')


#	for seed in ['1', '2', '3']:
#		to_zip = datadir + size + '/select' + select + '/' + arch + '/' + seed + '/checkpoints'
#		os.system('zip -r ' + to_zip + '.zip ' + to_zip)
#		os.system('rm -r ' + to_zip)


if __name__== '__main__':
	'''Command: python pathtothisfile pathtodatadir iteration/experiment
	e.g. python scripts/fq_transformer_wu.py /blue/liu.ying/al_morphseg/al_pilot_data/ lez50 transformer'''

	metrics = {}
	evaluation_file = sys.argv[1] + sys.argv[2] + '/' + sys.argv[3] + '/' + sys.argv[7] + '/' + sys.argv[4] + '/select' + sys.argv[5] + '/' + sys.argv[6] + '/eval.txt'

	if os.path.exists(evaluation_file) and os.stat(evaluation_file).st_size != 0:
		pass
	else:
		main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
		print('\n########### FINISHED ', sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7], ' ###########\n')
