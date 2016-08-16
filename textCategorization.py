
# coding: utf-8

# In[84]:

# imports
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
import os
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
import itertools
from nltk.tag import StanfordNERTagger
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
import tabulate


# In[68]:

# initialising and module function definitions
st = StanfordNERTagger("/Python/stanford-ner-2014-06-16/classifiers/english.all.3class.distsim.crf.ser.gz","//Python/stanford-ner-2014-06-16/stanford-ner.jar")
stemmer = PorterStemmer()

# opening file
def fileOpen(fileLocation):
    filePointer = open(fileLocation,"r").read()
    return filePointer

# NER calculator
def nerCalculator(finalCorpus):
    ner_corpus = []
    for i in finalCorpus:
        ner_list = st.tag(i.decode('utf-8').split())
        nerCollate = []
        for each_element in ner_list:
            if each_element[1]=="PERSON" or each_element[1]=="LOCATION" or                 each_element[1]=="ORGANIZATION" or each_element[1]=="TIME" or                 each_element[1]=="MONEY" or each_element[1]=="PERCENT" or                 each_element[1]=="DATE":
                    nerCollate.append(each_element[0])
        doc2 = " ".join(nerCollate)
        ner_corpus.append(doc2)
    return ner_corpus

# POS calculator
def posCalculator(finalCorpus):
    pos_corpus = []
    for i in finalCorpus:
        pos_list = nltk.pos_tag(word_tokenize(i.decode('utf-8')))
        posCollate = []
        for each_element in pos_list:
            #NOUN, ADJECTIVE AND VERB
            if each_element[1]=="NN" or each_element[1]=="NNS" or each_element[1]=="NNP" or                each_element[1]=="NNPS" or each_element[1]=="JJ" or each_element[1]=="JJR" or                each_element[1]=="JJS" or each_element[1]=="VB" or each_element[1]=="VBD" or                each_element[1]=="VBG" or each_element[1]=="VBZ" or each_element[1]=="VBN" or each_element[1]=="VBP":
               posCollate.append(each_element[0])
        doc1 = " ".join(posCollate)
        pos_corpus.append(doc1)
    return pos_corpus


# In[69]:

# reading each category and its documents
# category - WorldWar
stalin = fileOpen("//Documents/BA-BD/Sem-3/nlp/categories/worldwar/stalin.txt")
franklin = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/worldwar/franklingroosewelt.txt")
musolini = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/worldwar/musolini.txt")
kaishek = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/worldwar/kaishek.txt")
octoberRev = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/worldwar/octoberrevolution.txt")

# category - marketing
afmarketing = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/marketing/affiliatemarketing.txt")
digmarketing = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/marketing/digitalmarketing.txt")
marketing = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/marketing/marketing.txt")
marketingmanagement = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/marketing/marketingmanagement.txt")
strmanagement = fileOpen("//Documents/BA-BD/Sem-3/nlp/categories/marketing/strategicmanagement.txt")

# category - programming
compiler = fileOpen("//Documents/BA-BD/Sem-3/nlp/categories/programming/compiler.txt")
debugging = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/programming/debugging.txt")
design = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/programming/design.txt")
programming = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/programming/programming.txt")
testing = fileOpen("// /Documents/BA-BD/Sem-3/nlp/categories/programming/testing.txt")

# final corpus collation
finalCorpus = [afmarketing,digmarketing,marketing,marketingmanagement,strmanagement,
              compiler,debugging,design,programming,testing,
               stalin,franklin,musolini,kaishek,octoberRev]


# # Calculating weights
# Here I have used TF-IDF weights to calculate the cosine similarity of documents.
# Then a comparison of first document to all other documents are made. After the metric has been created, a k-means clustering is done to gather similar documents together. From the cluster table its is clear that documents of same category are floaking together.

# In[73]:

# calculating tf-idf weights
tfVec = TfidfVectorizer()
finalCorpusTfidf = tfVec.fit_transform(finalCorpus).todense()

# calculating cosine similarity
metric = []
cosSimilarity = linear_kernel(finalCorpusTfidf[0:1], finalCorpusTfidf).flatten()
metric.append(cosSimilarity)

# getting the labels
labels = ["afmarketing","digmarketing","marketing","marketingmanagement","strmanagement",
          "compiler","debugging","design","programming","testing",
          "stalin","franklin","musolini","kaishek","octoberRev"]


# In[74]:

# K-means for Cosine Similarity
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(finalCorpusTfidf)
clusters_COS = km.labels_.tolist()


# In[88]:

# table with CLUSTERS
dictio3 = {'Labels':labels,'Clusters':clusters_COS}
frame_Cosine = pd.DataFrame(dictio3)
print tabulate.tabulate(frame_Cosine, headers= dictio3.keys(), tablefmt='psql')


# # Calculating weights using POS tags
# Here also the clustering is done using metric - cosine value using POS tags. The result shows a slight misprediction of "strmanagement" document. 

# In[76]:

# pos calculation
pos_corpus = posCalculator(finalCorpus)
# tf-idf calculation for POS
posTFIDF = tfVec.fit_transform(pos_corpus).todense()

# calculating cosine similarity for POS
metric_pos = []
cosSimilarity_pos = linear_kernel(posTFIDF[0:1], posTFIDF).flatten()
metric_pos.append(cosSimilarity_pos)
print metric_pos


# In[77]:

# charting the metrics
df = pd.DataFrame(data=metric_pos,index=labels,columns=labels)
# K-means for POS
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(posTFIDF)
clusters_POS = km.labels_.tolist()


# In[89]:

# table with CLUSTERS
dictio2 = {'Labels':labels,'Clusters':clusters_POS}
frame_POS = pd.DataFrame(dictio2)
print tabulate.tabulate(frame_POS, headers= dictio2.keys(), tablefmt='psql')


# # Calculation of clusters using NER tags
# The result shows that using similarity using NER tags is not at all a good measure of finding similar documents. The documents under consideration does not have enough NER tags for proper clustering of documents.

# In[79]:

# NER Calculation
ner_corpus = nerCalculator(finalCorpus)
# tf-idf for NER
nerTFIDF = tfVec.fit_transform(ner_corpus).todense()
# calculating cosine similarity
metricNER = []
cosSimilarity_ner = linear_kernel(nerTFIDF[0:1], nerTFIDF).flatten()
metricNER.append(cosSimilarity_ner)
print metricNER


# In[80]:

# using k-means for NER
num_clusters = 3
km = KMeans(n_clusters=num_clusters)
km.fit(nerTFIDF)
clusters = km.labels_.tolist()


# In[90]:

dictio = {'Labels':labels,'Clusters':clusters}
frame = pd.DataFrame(dictio)
print tabulate.tabulate(frame, headers= dictio.keys(), tablefmt='psql')




