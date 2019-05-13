import re, collections, copy, random, string, pickle, json, sys, os, argparse
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.distributions import Bernoulli

sys.path.append('/home/aissa/miniconda3/lib/python3.6/site-packages/lucene-6.4.1-py3.6-linux-x86_64.egg')

import lucene, pytrec_eval
from lucene import *
from java.nio.file import Paths
from org.apache.lucene.analysis.standard import StandardAnalyzer, ClassicAnalyzer
from org.apache.lucene.analysis.en import EnglishAnalyzer, PorterStemFilter, KStemFilter
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.analysis.core import StopFilter
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.store import SimpleFSDirectory
from org.apache.lucene.search import IndexSearcher
from lucene.collections import JavaSet
from org.apache.lucene.analysis.util import CharArraySet

lucene.initVM(vmargs=['-Djava.awt.headless=true'])
analyzer = EnglishAnalyzer()

USE_CUDA = True

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="model")
    '''
    parser.add_argument(
        '--dataset', '-d', #type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="dataset path")
    
    
    parser.add_argument(
        '--save', '-o', #type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="save DIR path")
    
    parser.add_argument(
        '--index', '-i', #type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="index path")
    
    parser.add_argument(
        '--data', '-data', #type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="data NL Q file path")
    
    parser.add_argument(
        '--qrel', '-q', #type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="qrel file path")
    
    parser.add_argument(
        '--ids', '-ids', #type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="ids path")
    '''
    parser.add_argument(
        '--id', '-id' ,type=int, #default=2, metavar='FREQ',
        help='number of id to get for cross val between 1 and 10')
    
    return parser

def run(searcher, analyzer, evaluator, commands):
    res = {}
    for command in commands:
        if command[1]:
            #print ("Searching for:", command)
            query = QueryParser("content", analyzer).parse(command[1])
            scoreDocs = searcher.search(query, 1000).scoreDocs
            #print ("%s total matching documents." % len(scoreDocs))
            res[str(command[0])] = {}
            for i, scoreDoc in enumerate(scoreDocs):
                doc = searcher.doc(scoreDoc.doc)
                res[str(command[0])][doc.get("docno").strip()] = scoreDoc.score
        else: 
            res[str(command[0])] = {'map': 0.0}
            
    mapP = evaluator.evaluate(res)
    return mapP


def baseline(searcher, analyzer, evaluator, datafile, ids):
    
    with open(datafile) as file:
        
        data = file.read().split('\n')
        nlqs = [[d.split(' | ')[0], d.split(' | ')[2]] for d in data]
        qs = [[d.split(' | ')[0], d.split(' | ')[1]] for d in data]

    trainNL = [nl for nl in nlqs if nl[0] in ids['train']]
    trainQ = [q for q in qs if q[0] in ids['train']]

    testNL = [nl for nl in nlqs if nl[0] in ids['test']]
    testQ = [q for q in qs if q[0] in ids['test']]
    
    trainBIN = [[nl[0],' '.join([w for w in nl[1].split() if w in q[1].split()])] for nl,q in zip(nlqs, qs) if nl[0] in ids['train']]
    testBIN = [[nl[0],' '.join([w for w in nl[1].split() if w in q[1].split()])] for nl,q in zip(nlqs, qs) if nl[0] in ids['test']]

    res = run(searcher, analyzer, evaluator, trainNL)
    resTrainNL = np.mean([a['map'] for a in res.values()])

    res = run(searcher, analyzer, evaluator, trainQ)
    resTrainQ = np.mean([a['map'] for a in res.values()])

    res = run(searcher, analyzer, evaluator, testNL)
    resTestNL = np.mean([a['map'] for a in res.values()])

    res = run(searcher, analyzer,evaluator, testQ)
    resTestQ = np.mean([a['map'] for a in res.values()])
    
    res = run(searcher, analyzer,evaluator, trainBIN)
    resTrainBIN = np.mean([a['map'] for a in res.values()]) 
    
    res = run(searcher, analyzer,evaluator, testBIN)
    resTestBIN = np.mean([a['map'] for a in res.values()])   

    return resTrainNL, resTrainQ, resTestNL, resTestQ, resTrainBIN, resTestBIN


def getSample(size, dataset):
    
    #print(dataset)
    #print(batchSize)
    samples = copy.deepcopy(random.sample(dataset,size))
    #samples = copy.deepcopy(dataset[s:s+batchSize])
    #print(samples)
            
    orderedSamples = sorted(samples, key=lambda p: len(p[1]), reverse=True)
    input_lengths = [len(s[1]) for s in orderedSamples]
    ids , inputs, targets = zip(*orderedSamples)
    [sample.extend([0]*(max(input_lengths) - len(sample))) for sample in inputs]
    [sample.extend([0]*(max(input_lengths) - len(sample))) for sample in targets]
            
    inputVar = Variable(torch.LongTensor(inputs)).transpose(0, 1)
    targetVar = Variable(torch.FloatTensor(targets)).transpose(0, 1)

    if USE_CUDA:
        inputVar = inputVar.cuda()
        targetVar = targetVar.cuda()
        
    return ids, inputVar, input_lengths, targetVar

class Encoder(nn.Module):
    
    def __init__(self, vocabSize, hiddenSize, weights, n_layers=1):
        nn.Module.__init__(self)
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.lstm = nn.LSTM(weights.size(1), hiddenSize, n_layers, bidirectional=True)
        self.n_layers = n_layers
    
    
    def forward(self, inputs, input_lengths):
        embedded = self.embedding(inputs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, (h, c) = self.lstm(packed, None)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        h = h[:self.n_layers, :, :] + h[self.n_layers:, :, :]
        c = c[:self.n_layers, :, :] + c[self.n_layers:, :, :]
        
        return h, c
    
class Decoder(nn.Module):
    
    def __init__(self, vocabSize, hiddenSize, weights, n_layers=1):
        nn.Module.__init__(self)
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.embedding = nn.Embedding.from_pretrained(weights)
        self.W1 = nn.Linear(300,300)
        self.W2 = nn.Linear(hiddenSize,300)
        self.lstm0 = nn.LSTM(300 , hiddenSize , n_layers)
        self.lstm1 = nn.LSTM(300 , hiddenSize , n_layers)
        self.linear = nn.Linear(hiddenSize , 1)
        self.linear.weight.data.normal_(0.0, 1)
        
    def forward(self, x, h, c, hn, y):
        embedded = self.embedding(x).view(1, -1, 300) # 1 x B x N
        inputt = F.tanh(self.W1(embedded) + self.W2(hn))
        output, (h, c) = (1-y) * self.lstm0(inputt, (h, c)) + y * self.lstm1(inputt, (h, c))
        output = F.sigmoid(self.linear(output.squeeze(0)))
        return output, h, c

def beamSearch(h, c, hn, decoder, inputVar):
    
    output, h, c = decoder(inputVar[0].view(-1,1), h, c, hn, y=0)
    bestk = [[h, c, [1], output , torch.log(output)], [h, c, [0], 1-output , torch.log(1-output)]]
    
    for i in range(1, len(inputVar)):
        #print("input", inputVar[i])
        if (inputVar[i] == 0).item(): break
        candidates = []
        
        for best in bestk:
            h, c, seq, outputs, score = best
            
            output, h, c = decoder(inputVar[i].view(-1,1), h, c, hn, y=seq[-1])
                
            candidate0 = [h, c, seq + [0], torch.cat((outputs, 1-output),1), score + torch.log(1-output)]
            candidate1 = [h, c, seq + [1], torch.cat((outputs, output),1), score + torch.log(output)]
                
            candidates.append(candidate0)
            candidates.append(candidate1)
            
        ordered = sorted(candidates, key=lambda p: p[4], reverse=True)
        
        if len(bestk) > 4:
            bestk = ordered[:5]
        else:
            bestk = ordered
    
    return bestk[0]

def train(encoder, decoder, dataset, batchSize, loss='RL', lastReward=0.0):
    
    encoder.train(True)
    decoder.train(True)
    
    mapps = 0
    losses = 0.0
    
    for s in range(0, len(dataset)//batchSize):
        
        preds = []
        ids, inputVar, input_lengths, targetVar = getSample(batchSize, dataset)
        #print(ids)
        #print(targetVar.transpose(1,0))
        predicted = [['0'] * input_lengths[i] for i in range(batchSize)]
        encoderH, encoderC = encoder(inputVar, input_lengths) 
        if loss == 'NLL': 
            lossNLL = Variable(torch.zeros(batchSize))
            if USE_CUDA:
                lossNLL = lossNLL.cuda()
        else:
            lossRL = Variable(torch.zeros(batchSize))
            if USE_CUDA:
                lossRL = lossRL.cuda()
                            
        for b in range(batchSize):
            hn = encoderH[:,b].view(1, 1, 100)
            h = torch.zeros([1,1,100], dtype=torch.float)
            c = torch.zeros([1,1,100], dtype=torch.float)
            
            if USE_CUDA:
                h = h.cuda()
                c = c.cuda()
            
            previous = 0 
            for i in range(len(inputVar)):
                if (inputVar[i][b] == 0).item(): break
                output, h, c = decoder(inputVar[i][b].view(-1,1), h, c, hn, y=previous)
                m = Bernoulli(output)
                action = m.sample()
                
                if action.item() == 1:
                    predicted[b][i] = index2word[inputVar[i][b].item()]
                
                if loss == 'NLL':  
                    previous = int(targetVar[i][b].item())
                    lossNLL[b] = lossNLL[b] + criterion(output,targetVar[i][b].view(-1, 1))
                    
                else:
                    previous = int(action.item())
                    lossRL[b] = lossRL[b] + m.log_prob(action).squeeze()
                    
        maps = {}    
        for id, pr in zip(ids, predicted):
            preds.append([int(id), ' '.join([p for p in pr if p.isalpha()])])
        #print(predicted)     
        mapP = run(searcher, analyzer, evaluator, preds)    
        
        if loss == 'NLL':
            lossNLL.mean().backward()
            
        else:
            error = - lossRL * (Variable(torch.FloatTensor([a['map'] for a in mapP.values()]) - lastReward).cuda())  
            error.mean().backward()
            #print(error)
        
        #torch.nn.utils.clip_grad_norm_(encoder.parameters(), 0.5)
        #torch.nn.utils.clip_grad_norm_(decoder.parameters(), 0.5)
            
        optimizerE.step()
        optimizerD.step()
                
        optimizerE.zero_grad()
        optimizerD.zero_grad()
            
        mapps += np.mean([a['map'] for a in mapP.values()])

        if loss == 'NLL':
            losses += lossNLL.mean()
        
    if loss == 'NLL':
        return losses / (s+1)
    
def evaluate(encoder, decoder, dataset):
    
    encoder.train(False)
    decoder.train(False)
    
    preds = []
    ids, inputVar, input_lengths, targetVar = getSample(len(dataset), dataset)
    predicted = [['0'] * input_lengths[i] for i in range(len(dataset))]
    encoderH, encoderC = encoder(inputVar, input_lengths) 
            
    for b in range(len(dataset)):
        hn = encoderH[:,b].view(1, 1, 100)
        h = torch.zeros([1,1,100], dtype=torch.float)
        c = torch.zeros([1,1,100], dtype=torch.float)
            
        if USE_CUDA:
            h = h.cuda()
            c = c.cuda()
                   
        h, c, seq, outputs, score = beamSearch(h, c, hn, decoder, inputVar[:,b])
            
        for i in range(len(seq)):
            if seq[i]==1: predicted[b][i] = index2word[inputVar[i][b].item()]                 
  
    maps = {}    
    for id, pr in zip(ids, predicted):
        preds.append([int(id), ' '.join([p for p in pr if p.isalpha()])])
            
    mapP = run(searcher, analyzer, evaluator, preds)

    return np.mean([a['map'] for a in mapP.values()])


parser = create_parser()
args = parser.parse_args()

with open('/local/aissa/tradRL/data2/dataset2', 'rb') as handle:
    embeddings, indexes, dataset = pickle.load(handle)

index2word = {v: k for k, v in indexes.items()}
weights = torch.FloatTensor(embeddings)

datafile = '/local/aissa/tradRL/data2/data2.txt'

#datafile = args.data

qrel = {}
with open('/local/aissa/tradRL/data2/qrel2.txt', 'r') as file:
#with open(args.qrel, 'r') as file:
    for line in file:
        line = line.strip().split()
        if line[0] not in qrel.keys(): qrel[line[0]] = {}
        qrel[line[0]][line[2].lower()] = int(line[3])
        
#INDEX_DIR = args.index

INDEX_DIR = '/local/aissa/disk45_.index'

directory = SimpleFSDirectory(Paths.get(INDEX_DIR))
searcher = IndexSearcher(DirectoryReader.open(directory))
evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'map'})
with open('/local/aissa/tradRL/data2/ids', 'rb') as handle:
#with open(args.ids, 'rb') as handle:
    ids = pickle.load(handle)
ids = ids['ids'+str(args.id)]

trainSet = [d for d in dataset if d[0] in ids['train']]
testSet = [d for d in dataset if d[0] in ids['test']]


trainPlot, testPlot = [], []
trainPlotLoss, testPlotLoss = [], []
trainMaps, testMaps = 0, 0
trainLoss, testLoss = 0, 0 
save_every = 100
plot_every = 10
vocabSize = weights.size(0)
hiddenSize = 100
batchSize = 12
criterion = nn.BCELoss(reduce=True)

encoder = Encoder(vocabSize, hiddenSize, weights, 1)
decoder = Decoder(vocabSize, hiddenSize, weights, 1)

if USE_CUDA:
    encoder.cuda()    
    decoder.cuda()
    
lastReward = 0.0
epoch = 0

resTrainNL, resTrainQ, resTestNL, resTestQ, resTrainBIN, resTestBIN = baseline(searcher, analyzer, evaluator, datafile, ids)

optimizerE = torch.optim.Adam(filter(lambda p: p.requires_grad, encoder.parameters()), lr=0.01)
optimizerD = torch.optim.Adam(filter(lambda p: p.requires_grad, decoder.parameters()), lr=0.01)

#for epoch in range(101):
while round(lastReward,5) < round(resTrainBIN,5) and epoch<50:

    trainLoss = train(encoder, decoder, trainSet, batchSize, 'NLL')
    lastReward = evaluate(encoder, decoder, trainSet)
    trainMaps = lastReward
    testMaps = evaluate(encoder, decoder, testSet)
    
    trainPlotLoss.append(trainLoss)
    trainPlot.append(trainMaps)
    testPlot.append(testMaps)
    
    epoch += 1
    
fig = plt.figure()    
plt.plot(trainPlotLoss, label = 'train')
plt.legend()
plt.savefig("/net/big/aissa/data2/model12/save"+str(args.id)+"/lossNLL.png")
plt.close()
        
fig = plt.figure()    
plt.plot(trainPlot, label = 'train')
plt.plot(testPlot, label = 'test')
plt.axhline(y=resTrainNL, label = 'TrainNL', color = 'r')
plt.axhline(y=resTrainQ, label = 'TrainQ', color = 'm')
plt.axhline(y=resTestNL, label = 'TestNL', color = 'y')
plt.axhline(y=resTestQ, label = 'TestQ', color = 'k')
plt.axhline(y=resTestBIN, label = 'TestBIN', color = 'brown')
plt.axhline(y=resTrainBIN, label = 'TrainBin', color = 'tan')
plt.legend()
plt.savefig("/net/big/aissa/data2/model12/save"+str(args.id)+"/mapsNLL.png")
plt.close()
with open("/net/big/aissa/data2/model12/save"+str(args.id)+'/mapsNLL', 'wb') as fp:
    pickle.dump([trainPlot, testPlot], fp)
            
torch.save(encoder.state_dict(), "/net/big/aissa/data2/model12/save"+str(args.id)+'/encoder%dNLL.pth' % (epoch))
torch.save(decoder.state_dict(), "/net/big/aissa/data2/model12/save"+str(args.id)+'/decoder%dNLL.pth' % (epoch))
    
trainMaps, testMaps = 0, 0
trainPlot, testPlot = [], []

print('START RL', epoch)

optimizerE = torch.optim.SGD(filter(lambda p: p.requires_grad, encoder.parameters()), lr=0.003)
optimizerD = torch.optim.SGD(filter(lambda p: p.requires_grad, decoder.parameters()), lr=0.003)


for epoch in range(1001):

    lastReward = evaluate(encoder, decoder, trainSet)
    train(encoder, decoder, trainSet, batchSize, 'RL', lastReward)
    trainMaps += lastReward
    testMaps += evaluate(encoder, decoder, testSet)
                
    if epoch != 0 and epoch % plot_every == 0:
        trainPlot.append(trainMaps / plot_every)
        testPlot.append(testMaps / plot_every)
        trainMaps, testMaps = 0, 0

        fig = plt.figure()    
        plt.plot(trainPlot, label = 'train')
        plt.plot(testPlot, label = 'test')
        plt.axhline(y=resTrainNL, label = 'TrainNL', color = 'r')
        plt.axhline(y=resTrainQ, label = 'TrainQ', color = 'm')
        plt.axhline(y=resTestNL, label = 'TestNL', color = 'y')
        plt.axhline(y=resTestQ, label = 'TestQ', color = 'k')
        plt.axhline(y=resTestBIN, label = 'TestBIN', color = 'brown')
        plt.axhline(y=resTrainBIN, label = 'TrainBin', color = 'tan')
        plt.legend()
        plt.savefig("/net/big/aissa/data2/model12/save"+str(args.id)+"/maps.png")
        plt.close()
        with open("/net/big/aissa/data2/model12/save"+str(args.id)+'/maps', 'wb') as fp:
            pickle.dump([trainPlot, testPlot], fp)
            
    if epoch != 0 and epoch % save_every == 0 :
        torch.save(encoder.state_dict(), "/net/big/aissa/data2/model12/save"+str(args.id)+'/encoder%d.pth' % (epoch))
        torch.save(decoder.state_dict(), "/net/big/aissa/data2/model12/save"+str(args.id)+'/decoder%d.pth' % (epoch))
