import os
import conll18_ud_eval
import sys
import urllib.request as urllib
import json
import _jsonnet
import random

random.seed(8446)
seeds = ['1', '2', '3', '4', '5']

trains = ['UD_English-EWT', 'UD_Russian-Taiga', 'UD_Italian-PoSTWITA', 'UD_Russian-GSD', 'UD_Italian-ISDT', 'UD_English-WSJ']
all_treebanks = trains + ['UD_English-Atis', 'UD_English-ESL', 'UD_Naija-NSC']
treebank2lang = {'UD_English-Tweebank2':'en', 'UD_Italian-PoSTWITA':'it', 'UD_Naija-NSC':'pcm', 'UD_Norwegian-NynorskLIA': 'nn', 'UD_Slovenian-SSJ': 'sl'}

crossLing = {'UD_English-EWT': ['UD_Russian-Taiga', 'UD_Italian-PoSTWITA'], 'UD_Russian-Taiga': ['UD_English-EWT', 'UD_Italian-PoSTWITA'], 'UD_Italian-PoSTWITA': ['UD_English-EWT', 'UD_Russian-Taiga'], 'UD_Russian-GSD': ['UD_Italian-ISDT', 'UD_English-WSJ'], 'UD_Italian-ISDT': ['UD_Russian-GSD', 'UD_English-WSJ'], 'UD_English-WSJ': ['UD_Russian-GSD', 'UD_Italian-ISDT']}
crossDom = {'UD_English-EWT': ['UD_English-WSJ', 'UD_English-Atis', 'UD_English-ESL', 'UD_Naija-NSC'], 'UD_Russian-Taiga': ['UD_Russian-GSD'], 'UD_Italian-PoSTWITA': ['UD_Italian-ISDT'], 'UD_English-WSJ': ['UD_English-EWT', 'UD_English-Atis', 'UD_English-ESL', 'UD_Naija-NSC'], 'UD_Russian-GSD': ['UD_Russian-Taiga'], 'UD_Italian-ISDT': ['UD_Italian-PoSTWITA']}

devSizes = ['100', '200', '500', '1000', '2000', '5000', '10000', '20000', '9950000']
splitMethods = ['sequential', 'sequential_random', 'sequential_random_order']

multiRegressive = ['Helsinki-NLP/opus-mt-mul-en', 'bigscience/bloom-560m', 'facebook/mbart-large-50', 'facebook/mbart-large-50-many-to-many-mmt', 'facebook/mbart-large-50-many-to-one-mmt', 'facebook/mbart-large-50-one-to-many-mmt', 'facebook/mbart-large-cc25', 'facebook/mgenre-wiki', 'facebook/nllb-200-distilled-600M', 'facebook/xglm-564M', 'google/byt5-base', 'google/byt5-small', 'google/canine-c', 'google/canine-s', 'google/mt5-base', 'google/mt5-small']
multiAutoencoder = ['Peltarion/xlm-roberta-longformer-base-4096', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'cardiffnlp/twitter-xlm-roberta-base', 'distilbert-base-multilingual-cased', 'google/rembert', 'microsoft/infoxlm-base', 'microsoft/infoxlm-large', 'microsoft/mdeberta-v3-base', 'setu4993/LaBSE', 'studio-ousia/mluke-base', 'studio-ousia/mluke-base-lite', 'studio-ousia/mluke-large', 'studio-ousia/mluke-large-lite', 'xlm-mlm-100-1280', 'xlm-roberta-base', 'xlm-roberta-large'] 
mlms = multiRegressive + multiAutoencoder

diaparser_mlms = ["bert-base-multilingual-cased", "xlm-roberta-large",  "cardiffnlp/twitter-xlm-roberta-base", 'microsoft/infoxlm-large', 'Peltarion/xlm-roberta-longformer-base-4096', "xlm-roberta-base", 'bert-base-multilingual-uncased']
diaparser_mlms_map = {"bert-base-multilingual-cased": 'mbert', "xlm-roberta-large": 'xlmr_large',  "cardiffnlp-twitter-xlm-roberta-base": 'twitter_xlmr', 'microsoft-infoxlm-large': 'infoxlm', 'Peltarion-xlm-roberta-longformer-base-4096': 'longformer', "xlm-roberta-base": 'xlmr', 'bert-base-multilingual-uncased': 'mbert_uncased'}

def getEmbedSize(name):
    with urllib.urlopen("https://huggingface.co/" + name + "/raw/main/config.json") as url:
        data = json.loads(url.read().decode())
        for key in ['hidden_size', 'dim']:
            if key in data:
                return data[key]
    return 0

def load_json(path: str):
    """
    Loads a jsonnet file through the json package and returns a dict.
    
    Parameters
    ----------
    path: str
        the path to the json(net) file to load
    """
    return json.loads(_jsonnet.evaluate_snippet("", '\n'.join(open(path).readlines())))

def makeParams(mlm):
    config = load_json('configs/params.json')
    config['transformer_model'] = mlm
    tgt_path = 'configs/params.' + mlm.replace('/', '_') + '.json'
    if not os.path.isfile(tgt_path):
        json.dump(config, open(tgt_path, 'w'), indent=4)
    return tgt_path

def read_conllu(path):
    data = []
    curSent = []
    for line in open(path):
        if len(line) < 2:
            if curSent not in data:
                data.append(curSent)
            curSent = []
        else:
            if line[0] == '#':
                continue
            curSent.append(line.strip().split('\t'))
    return data


def getMachampModel(name):
    modelDir = 'machamp/logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.pt'
            if os.path.isfile(modelPath):
                return modelPath
    return ''

def getTrainDevTest(path):
    train = ''
    dev = ''
    test = ''
    for conlFile in os.listdir(path):
        if conlFile.endswith('conllu'):
            if 'train' in conlFile:
                train = path + '/' + conlFile
            if 'dev' in conlFile:
                dev = path + '/' + conlFile
            if 'test' in conlFile:
                test = path + '/' + conlFile
    return train, dev, test

def genDataconfig(treebank):
    outPath = 'configs/' + treebank + '.json'
    if os.path.isfile(outPath):
        return outPath
    trainPath = 'data/' + treebank + '/train.conllu'
    base = load_json('machamp/configs/ewt.json')
    new = {treebank:base['UD_EWT']}
    new[treebank]['train_data_path'] = '../' + trainPath
    del new[treebank]['dev_data_path']
    for task in ['feats', 'lemma', 'upos']:
        del new[treebank]['tasks'][task]
    json.dump(new, open(outPath, 'w'), indent = 4)
    return outPath

def getSplit(lengths, splitMethod, max_size):
    if max_size == 9950000:
        return range(0, len(lengths))

    elif splitMethod == 'sequential':
        # get last sequence of sentences that fits in max_size
        size = 0
        beg_sent_idx = len(lengths)-1
        while size + lengths[beg_sent_idx] <= max_size or size == 0:
            size += lengths[beg_sent_idx]
            beg_sent_idx -= 1
        return range(beg_sent_idx, len(lengths))

    elif splitMethod == 'sequential_random':
        # get random list of instances that fit in max_size
        # we do this by shuffling a list of indices, and picking
        # the top-n
        indices = list(range(len(lengths)))
        random.shuffle(indices)
        size = 0
        end_sent_idx = 0
        while size + lengths[indices[end_sent_idx]] <= max_size or size == 0:
            size += lengths[indices[end_sent_idx]]
            end_sent_idx += 1
        return indices[:end_sent_idx]
        
    elif splitMethod == 'sequential_random_order':
        # find last possible sequence
        size = 0
        beg_sent_idx = len(lengths)-1
        while size + lengths[beg_sent_idx] <= max_size or size == 0:
            size += lengths[beg_sent_idx]
            beg_sent_idx -= 1

        # start should be in-between 0 and last possible sequence
        beg = random.randint(0,beg_sent_idx)
        # find end
        size = 0
        end_sent_idx = beg
        while size + lengths[end_sent_idx] <= max_size or size == 0:
            size += lengths[end_sent_idx]
            end_sent_idx += 1
        return range(beg, end_sent_idx+1)
        

