import pandas as pd
import sys

result_file = sys.argv[1]

descriptive_stats = pd.read_csv("ud_overall_stats.csv")
treebanks = descriptive_stats['treebank'].tolist()
lang_families = descriptive_stats['Language Family'].tolist()

treebank_lang_family_dict = {}
for i in range(len(treebanks)):
	treebank_lang_family_dict[treebanks[i]] = lang_families[i]

updated_data = []
with open(result_file) as f:
	for line in f:
		if line.startswith('Treebank'):
			header = line.strip().split()
			header.append('Language')
			header.append('Language_Family')
			updated_data.append(header)
		else:
			toks = line.strip().split()
			treebank = toks[0]
			try:
				lang_family = treebank_lang_family_dict[treebank]
				lang = treebank.split('-')[0].split('_')[1]
				toks.append(lang)
				toks.append(lang_family)
				updated_data.append(toks)
			except:
				print(treebank)

with open('new_' + result_file, 'w') as f:
	for tok in updated_data:
		f.write(' '.join(str(w) for w in tok) + '\n')