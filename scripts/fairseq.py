#!/usr/bin/env python
# coding: utf-8

import subprocess
import os
import sys, statistics
from datetime import datetime
from sklearn.metrics import precision_recall_fscore_support

# al_trainselect/btz_surSeg50/select0/transformer/1

def main(treebank, size, idx, select, arch, task = 'pos', method = 'al'):

	subprocess.run(['mkdir', '-p', task + '_data/' + method + '/' + treebank + '/train' + size])
	subprocess.run(['mkdir', '-p', task + '_data/' + method + '/' + treebank + '/train' + size + '/select' + select])
	subprocess.run(['mkdir', '-p', task + '_data/' + method + '/' + treebank + '/train' + size + '/select' + select + '/' + arch])

	sub_datadir = task + '_data/' + method + '/' + treebank + '/train' + size + '/select' + select + '/'
	previous_datadir = task + '_data/' + method + '/' + treebank + '/train' + size + '/select' + str(int(select) - 25) + '/'

	if select == '0':
		os.system('cp ' + task + '_data/' + method + '/' + treebank + '/' + 'train.' + size + '_' + idx + '.input ' + sub_datadir)
		os.system('cp ' + task + '_data/' + method + '/' + treebank + '/' + 'train.' + size + '_' + idx + '.output ' + sub_datadir)
		os.system('cp ' + task + '_data/' + method + '/' + treebank + '/' + 'select.' + size + '_' + idx + '.input ' + sub_datadir)
		os.system('cp ' + task + '_data/' + method + '/' + treebank + '/' + 'select.' + size + '_' + idx + '.output ' + sub_datadir)

	else:
		os.system('cat ' + task + '_data/' + method + '/' + treebank + '/' + 'train.' + size + '_' + idx + '.input ' + previous_datadir + '/increment.input >' + sub_datadir + 'train.' + size + '_' + idx + '.input')
		os.system('cat ' + task + '_data/' + method + '/' + treebank + '/' + 'train.' + size + '_' + idx + '.output ' + previous_datadir + '/increment.output >' + sub_datadir + 'train.' + size + '_' + idx + '.output')
		os.system('mv ' + previous_datadir + 'residual.input ' + sub_datadir + 'select.' + size + '_' + idx + '.input')
		os.system('mv ' + previous_datadir + 'residual.output ' + sub_datadir + 'select.' + size + '_' + idx + '.output')

	SRC = size + '_' + idx + '.input'
	TGT = size + '_' + idx + '.output'
	TESTIN = task + '_data/' + method + '/' + treebank + '/test.full.input'
	TESTIN_gold = task + '_data/' + method + '/' + treebank + '/test.full.output'
	TO_PREDICT = sub_datadir + 'select.'+ size + '_' + idx '.input'

	### Collecting the pool of words to select from
	select_input = []
	with open(sub_datadir + 'select.'+ size + '_' + idx + '.input') as f:
		for line in f:
			toks = line.strip()
			select_input.append(toks)

	select_output = []
	with open(sub_datadir + 'select.'+ size + '_' + idx + '.output') as f:
		for line in f:
			toks = line.strip()
			select_output.append(toks)

	select_combo = [select_input[i] + '_' + select_output[i] for i in range(len(select_input))]

	confidence_dict = {}

	for seed in ['1', '2', '3']:
		PREDDIR = task + '_data/' + method + '/' + treebank + '/train' + size + '/select' + select + '/' + arch + '/' + seed + '/preds/' ## seed
		FROMDIR = task + '_data/' + method + '/' + treebank + '/train' + size + '/select' + select + '/' + arch + '/' + seed + '/data-bin/'
		SAVEDIR = task + '_data/' + method + '/' + treebank + '/train' + size + '/select' + select + '/' + arch + '/' + seed + '/checkpoints/'
	#    SCOREDIR = datadir + experiment + '/' + task + '/' + arch + '/' + seed + '/scores/'
	
		MODELPATH = SAVEDIR + 'checkpoint_best.pt'   
	
		GUESSPATH = PREDDIR + 'test.' + size + '_' + idx + '.predict'
		U_CONFIDENCE = PREDDIR + size + '_' + idx + '.confidence'
		U_PREDICTIONS = PREDDIR + size + '_' + idx + '.predict'
	
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
						'--source-treebank=' + SRC,
						'--target-treebank=' + TGT,
						'--trainpref='+ sub_datadir + '/train',
						'--validpref='+ sub_datadir + '/train',
						#'--testpref='+TEXTDIR+'test',
						'--destdir=' + FROMDIR])
	
		print("##################### PREPROCESSED", treebank + ' ' + size + ' ' + idx, "#####################")
	
		# train
		#Transformer Parameters from Wu 2021 et al.: ACTIVATION_DROPOUT=0.3, ACTIVATION_FN='RELU', ADAM_BETAS='(0.9, 0.98)', ADAM_EPS=1E-08, ADAPTIVE_INPUT=FALSE, ADAPTIVE_SOFTMAX_CUTOFF=NONE, ADAPTIVE_SOFTMAX_DROPOUT=0, CROSS_SELF_ATTENTION=FALSE, DECODER_ATTENTION_HEADS=4, DECODER_EMBED_DIM=256, DECODER_EMBED_PATH=NONE, DECODER_FFN_EMBED_DIM=1024, DECODER_INPUT_DIM=256, DECODER_LAYERDROP=0, DECODER_LAYERS=4, DECODER_LEARNED_POS=FALSE, DECODER_NORMALIZE_BEFORE=TRUE, DECODER_OUTPUT_DIM=256, DISTRIBUTED_WRAPPER='DDP', DROPOUT=0.3, ENCODER_ATTENTION_HEADS=4, ENCODER_EMBED_DIM=256, ENCODER_FFN_EMBED_DIM=1024, ENCODER_LAYERDROP=0, ENCODER_LAYERS=4, ENCODER_LEARNED_POS=FALSE, ENCODER_NORMALIZE_BEFORE=TRUE, EVAL_BLEU=FALSE, EVAL_BLEU_ARGS=NONE, EVAL_BLEU_DETOK='SPACE', EVAL_BLEU_DETOK_ARGS=NONE, EVAL_BLEU_PRINT_SAMPLES=FALSE, EVAL_BLEU_REMOVE_BPE=NONE, EVAL_TOKENIZED_BLEU=FALSE, FIXED_VALIDATION_SEED=NONE, IGNORE_PREFIX_SIZE=0, LABEL_SMOOTHING=0.1, LAYERNORM_EMBEDDING=FALSE, LEFT_PAD_SOURCE='TRUE', LEFT_PAD_TARGET='FALSE', LOAD_ALIGNMENTS=FALSE, MAX_SOURCE_POSITIONS=1024, MAX_TARGET_POSITIONS=1024, NO_CROSS_ATTENTION=FALSE, NO_SCALE_EMBEDDING=FALSE, NO_TOKEN_POSITIONAL_EMBEDDINGS=FALSE, NUM_BATCH_BUCKETS=0, 
		# pipeline_balance=None, pipeline_decoder_balance=None, pipeline_decoder_devices=None, pipeline_encoder_balance=None, 
		#QUANT_NOISE_PQ=0, QUANT_NOISE_PQ_BLOCK_SIZE=8, QUANT_NOISE_SCALAR=0, REPORT_ACCURACY=FALSE, SHARE_ALL_EMBEDDINGS=FALSE, SHARE_DECODER_INPUT_OUTPUT_EMBED=TRUE, TIE_ADAPTIVE_WEIGHTS=FALSE, TRUNCATE_SOURCE=FALSE, UPSAMPLE_PRIMARY=1, USE_OLD_ADAM=FALSE=, WARMUP_INIT_LR=-1, WARMUP_UPDATES=1000, WEIGHT_DECAY=0.0)

		#subprocess.call(['./train.sh', dataset])
		subprocess.call(['fairseq-train', 
					 FROMDIR,
					'--save-dir=' + SAVEDIR,
					#'--restore-file=' + MODELPATH
					'--source-treebank=' + SRC,
					'--target-treebank=' + TGT,
					'--arch=' + arch,
					#'--pipeline-encoder-balance=4',
					#'--pipeline-decoder-balance=4',
					'--batch-size=400',
					#'--batch-size-valid=400',
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
					'--skip-invalid-size-inputs-valid-test'
					]) # not sure about these parameters to match Wu et al. 2020. 7/19/22
	
		print("##################### TRAINED", treebank + ' ' + size + ' ' + idx, "#####################")
	
		# test
		testing = subprocess.Popen(['fairseq-interactive',
							  FROMDIR,
							  '--path', MODELPATH,
							  '--source-treebank='+SRC,
							  '--target-treebank='+TGT,
							  '--skip-invalid-size-inputs-valid-test'
							 ],
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
		
		print("##################### TESTED", treebank + '_' + task + size, "#####################")
		
		# predict on unlabeled data and write predictions to file
		predicting = subprocess.Popen(['fairseq-interactive',
							  FROMDIR,
							  '--path', MODELPATH,
							  '--source-treebank='+SRC,
							  '--target-treebank='+TGT,
							  '--skip-invalid-size-inputs-valid-test'
							 ],
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
	
		print("##################### PREDICTED", treebank + '_' + task + size, "#####################")
		
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

	for k, v in confidence_dict.items():
		confidence_dict[k] = statistics.mean(v)
	
	sorted_confidence_dict = sorted(confidence_dict.items(), key = lambda item: item[1])
	increment_words = sorted_confidence_dict[ : 25]
	for combo in increment_words:
		print(combo[0], confidence_dict[combo[0]])
	print('')
	print('')
	for combo in sorted_confidence_dict[26 : 30]:
		print(combo[0], confidence_dict[combo[0]])

	increment_input = open(sub_datadir + 'increment.input', 'w')
	increment_output = open(sub_datadir + 'increment.output', 'w')
	increment_pairs = []
	for combo in increment_words:
		pair = combo[0]
		increment_pairs.append(pair)
		pair = pair.split('_')
		w_input = pair[0]
		w_output = pair[1]
		increment_input.write(w_input + '\n')
		increment_output.write(w_output + '\n')


	residual_input = open(sub_datadir + 'residual.input', 'w')
	residual_output = open(sub_datadir + 'residual.output', 'w')
	for pair in select_combo:
		if pair not in increment_pairs:
			pair = pair.split('_')
			w_input = pair[0]
			w_output = pair[1]
			residual_input.write(w_input + '\n')
			residual_output.write(w_output + '\n')

 
if __name__== '__main__':
	'''Command: python pathtothisfile pathtodatadir iteration/experiment
	e.g. python scripts/fq_transformer_wu.py /blue/liu.ying/al_morphseg/al_pilot_data/ lez50 transformer'''
	
	main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
	print('\n########### FINISHED ', sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], ' ###########\n')

#    files = [f for f in os.listdir(sys.argv[1]) if f.endswith('input') and f.startswith('train')]
#    for filename in files:
#        experiment = filename.split('.')[1]
#        if experiment == sys.argv[2]:
#            main(sys.argv[1], experiment, sys.argv[3])
#            print('\n########### FINISHED ', experiment, ' ###########\n')
