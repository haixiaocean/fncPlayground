import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora, models, similarities 
import re
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score

LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
RELATED = LABELS[0:3]

def getF1(gLabels,tLabels):
	return (f1_score(gLabels,tLabels))

#get the values for f1 score for related and unrelated articles
def getCosineThreshold(cosineSimMatrix,val):
	generateLabelsonThreshold = []
	generateOrigLabels = []

	for cosineDist, label in cosineSimMatrix:
		if cosineDist > val:
			generateLabelsonThreshold.append(1)
		else:
			generateLabelsonThreshold.append(0)
		if label in RELATED:
			generateOrigLabels.append(1)
		else:
			generateOrigLabels.append(0)

	return getF1(generateLabelsonThreshold,generateOrigLabels)

#read the stances and body

TrainStances = pd.read_csv('data/train_stances.csv')
TrainBodies = pd.read_csv('data/train_bodies.csv') 

#define stopWords and the lemmatizer
stop = stopwords.words('english') 
lemmatizer = WordNetLemmatizer()

#process the body text
bodyIds = TrainBodies['Body ID']
bodyText = TrainBodies['articleBody'] 

bodyText = np.array(bodyText)
bodyTextOrig = bodyText
bodyIds = np.array(bodyIds)
bodyText = [line.lower().split() for line in bodyText]
bodyText = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', '',word)) for word in text if word not in stop] for text in bodyText]

#create the dictionary for article bodies

bodyDict = dict(zip(bodyIds,bodyText))
bodyDictOrig = dict(zip(bodyIds,bodyTextOrig))
#create the dictionary of words

dictionary = corpora.Dictionary(bodyDict.values())
CorporaFNCbody = [dictionary.doc2bow(sent) for sent in bodyDict.values()]  
tfIdf = models.TfidfModel(CorporaFNCbody)

FncTfidf = tfIdf[CorporaFNCbody]
bodyDictString = dict(zip(bodyDict.keys(),range(0,len(bodyDict.values())))) 
#define LSI and number of topics (k)
lsi = models.LsiModel(FncTfidf, id2word=dictionary, num_topics=150)
#LSI matrix for FNC body corpus
FncLsi = lsi[FncTfidf]
#create the stance corpora 

sents = TrainStances['Headline']

sents = [sent.lower().split() for sent in sents]
#sentsOrig = sents
sents = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', '', w)) for w in sent if w not in stop] for sent in sents]


stanceDict = dict(zip(range(0,len(np.unique(sents))),np.unique(sents)))
#stanceDictOrig = dict(zip(range(0,len(np.unique(sentsOrig))),np.unique(sentsOrig)))
#also define the reverse stance dictionary
revStanceDict = { str(v): k for k, v in stanceDict.iteritems()}

stanceCorpora = [dictionary.doc2bow(t) for t in stanceDict.values()]
#print ("a sample stance corpora")
#print(stanceCorpora[0])

stanceTfidf = tfIdf[stanceCorpora]
#computing the LSI for stances
stanceLsi = lsi[stanceTfidf]
#create a list from stance and article body combination called miniList

TrainStances = np.array(TrainStances)
TrainStances[:,0] = [a.lower().split() for a in TrainStances[:,0]]
TrainStances[:,0] = [[lemmatizer.lemmatize(re.sub('[^a-zA-Z0-9]', '', w)) for w in sent if w not in stop] for sent in TrainStances[:,0]]
miniList = []
for i in range(0,len(TrainStances)):
	miniList.append([(revStanceDict[str(TrainStances[i][0])]),TrainStances[i][1],TrainStances[i][2]])


#compute the similarities of all sentences in the document bodies 

#sim = similarities.docsim.MatrixSimilarity(FncTfidf)
#Get similarities for LSI
sim = similarities.docsim.MatrixSimilarity(FncLsi)
#stance to body similarities

Stance2BodySim = [sim.get_similarities(stancetf) for stancetf in stanceLsi]

#compute cosine similarity for the milist
cosineSim = [[Stance2BodySim[lst[0]][bodyDictString[lst[1]]] , lst[2]] for lst in miniList]

#plot the histograms for cosine similarities

cosineSimUnrelated = []
for sim in cosineSim:
    if sim[1] == 'unrelated':
        cosineSimUnrelated.append(sim[0])

cosineSimRelated = []
for sim in cosineSim:
    if sim[1] != 'unrelated':
        cosineSimRelated.append(sim[0])


plt.hist(cosineSimRelated)
plt.xlabel("Cosine similarities histogram for related articles")
plt.savefig('plots/histCosineRelated.png')

plt.hist(cosineSimUnrelated)
plt.xlabel("Cosine similarities histogram comparing both related and unrelated articles")
plt.savefig('plots/histCosineBoth.png')

plt.clf()
#plot cosine threshold and f1 score
cosineThreshold = np.arange(0.0,1.0,0.01)
Y = [getCosineThreshold(cosineSim,a) for a in cosineThreshold]
plt.plot(cosineThreshold,Y)
plt.ylabel("F1 score on related/unrelated")
plt.xlabel("cosine threshold value")
plt.savefig('plots/f1Scores.png')

print ("Highest f1 score:" + str(np.max(Y)) + " for cosine threshold value of:"  + str(cosineThreshold[np.argmax(Y)]))
	
