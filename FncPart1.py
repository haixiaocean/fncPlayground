import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from gensim import corpora, models, similarities 
import re
import matplotlib.pyplot as plt

#read the stances and body

TrainStances = pd.read_csv('data/train_stances.csv')
TrainBodies = pd.read_csv('data/train_bodies.csv') 

#define stopWords
stop = stopwords.words('english') 

#process the body text
bodyIds = TrainBodies['Body ID']
bodyText = TrainBodies['articleBody'] 

bodyText = np.array(bodyText)
bodyIds = np.array(bodyIds)

bodyText = [line.lower().split() for line in bodyText]
bodyText = [[re.sub('[^a-zA-Z0-9]', '',word) for word in text if word not in stop] for text in bodyText]
#create the dictionary for article bodies

bodyDict = dict(zip(bodyIds,bodyText))

#create the dictionary of words

dictionary = corpora.Dictionary(bodyDict.values())
CorporaFNCbody = [dictionary.doc2bow(sent) for sent in bodyDict.values()]  
tfIdf = models.TfidfModel(CorporaFNCbody)

FncTfidf = tfIdf[CorporaFNCbody]
bodyDictString = dict(zip(bodyDict.keys(),range(0,len(bodyDict.values())))) 

#create the stance corpora 

sents = TrainStances['Headline']
sents = [sent.lower().split() for sent in sents]
sents = [[re.sub('[^a-zA-Z0-9]', '', w) for w in sent if w not in stop] for sent in sents]

stanceDict = dict(zip(range(0,len(np.unique(sents))),np.unique(sents)))
#also define the reverse stance dictionary
revStanceDict = { str(v): k for k, v in stanceDict.iteritems()}

stanceCorpora = [dictionary.doc2bow(t) for t in stanceDict.values()]
print ("a sample stance corpora")
print(stanceCorpora[0])

stanceTfidf = tfIdf[stanceCorpora]

#create a list from stance and article body combination called miniList

TrainStances = np.array(TrainStances)
miniList = []
for i in range(0,len(TrainStances)):
	miniList.append([(revStanceDict[str(sents[i])]),TrainStances[i][1],TrainStances[i][2]])


#compute the similarities of all sentences in the document bodies 

sim = similarities.docsim.MatrixSimilarity(FncTfidf)

#compute cosine similarity for the milist
cosineSim = [[sim[stanceTfidf[lst[0]]][bodyDictString[lst[1]]] , lst[2]] for lst in miniList]

#plot the histograms for cosine similarities

cosineSimUnrelated = []
for sim in cosineSim:
    if sim[1] == 'unrelated' and sim[0] > 0.0:
        cosineSimUnrelated.append(sim[0])

cosineSimRelated = []
for sim in cosineSim:
    if sim[1] != 'unrelated' and sim[0] > 0.0:
        cosineSimRelated.append(sim[0])


plt.hist(cosineSimRelated)
plt.ylabel("Histogram for cosine similarities for related articles")
plt.savefig('plots/histCosineRelated.png')

plt.hist(cosineSimUnrelated)
plt.ylabel("Histogram for cosine similarities for unrelated articles")
plt.savefig('plots/histCosineUnrelated.png')

