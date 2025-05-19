#import myutils
import os
import math
import matplotlib.pyplot as plt

plt.style.use('scripts/nice.mplstyle')

def getNumWords(path):
    if path == '':
        return 0
    counter = 0
    for line in open(path):
    #    if len(line) < 2: # for sents
        if line[0] in '0123456789':
            counter += 1
    return counter

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

fig, ax = plt.subplots(figsize=(8,5), dpi=300)

train_size_stats = open('ud_train_size_stats.txt', 'w')
header = ['Treebank', 'Train_size']
train_size_stats.write('\t'.join(header) + '\n')

datasets = {}
dataDir = 'ud-treebanks-v2.14/'
trainSizes = []
testSizes = []
for treebankDir in os.listdir(dataDir):
    if not treebankDir.startswith('UD'):
        continue
    train,dev,test = getTrainDevTest(dataDir + treebankDir)
    trainWords = getNumWords(train)
    testWords = getNumWords(test)
    trainSizes.append(trainWords)
#    testSizes.append(testWords)
    train_size_stats.write('\t'.join([treebankDir, str(trainWords)]) + '\n')

#ax.plot(sorted(devSizes), range(len(devSizes)), label='dev')
#ax.plot(sorted(testSizes), range(len(testSizes)), label='test')
ax.plot(range(len(trainSizes)), sorted(trainSizes), label='train')
#ax.plot(range(len(testSizes)), sorted(testSizes), label='test')

print(sorted(trainSizes))
#print(sorted(testSizes))

#ax.set_xlim((0,50000))
#ax.set_ylim((0,220))
#ax.set_ylim((0,2844511))
ax.set_ylim((0, 1000000))
ax.set_xlim((0,283))
ax.set_ylabel('Size (#tokens)')
ax.set_xlabel('#treebanks')
leg = ax.legend()
leg.get_frame().set_linewidth(1.5)

fig.savefig('datasizes.pdf', bbox_inches='tight')
