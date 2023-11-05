from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
import nltk
nltk.download('popular')

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from collections import Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import math
import operator
from sklearn.preprocessing import normalize
import numpy as np
import string
import pandas as pd
import collections
from numpy import linalg as LA
import re
import numpy as np

"""# ***read all decuments manually and store each doc in one specific variable***"""

# read all these files from google drive
d1 = open("/content/drive/MyDrive/data/1.txt")
d2 = open("/content/drive/MyDrive/data/2.txt")
d3 = open("/content/drive/MyDrive/data/3.txt")
d4 = open("/content/drive/MyDrive/data/4.txt")
d5 = open("/content/drive/MyDrive/data/5.txt")
d6 = open("/content/drive/MyDrive/data/6.txt")
d7 = open("/content/drive/MyDrive/data/7.txt")
d8 = open("/content/drive/MyDrive/data/8.txt")
d9 = open("/content/drive/MyDrive/data/9.txt")
d10 = open("/content/drive/MyDrive/data/10.txt")

"""# ***import os to access data folder in drive***
# ***import glob to access all files inside data folder***
"""

import os
file_location_input = os.path.join('/content', 'drive','MyDrive', 'data','*.txt')
print(file_location_input)

import glob
filenamesin = glob.glob(file_location_input)
print(filenamesin)

"""# ***read all files (using for loop) before preprocessing, each line is considered as one file***"""

#read all files before
for f in filenamesin:
    readfile = open(f,'r')
    data = readfile.readlines()

    readfile.close()
    print(data)

"""# **read all files use for loop after doing preprocessing ( tokenizion and remove punctuation,remove stop words,lemmatization words,stemming words)**"""

#read all files after

for f in filenamesin:
    readfile= open(f,'r')
    data = readfile.readlines()
    readfile.close()
    for line in data:
      lower = line.lower()
      tokenizer = nltk.RegexpTokenizer(r"\w+")
      new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

      stopword = stopwords.words('english')
      removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

      wordnet_lemmatizer = WordNetLemmatizer()
      lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

      wordnet_stemming = PorterStemmer()
      stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

      print(lemmatized_word)

"""# **read each file one by one and doing preprocessing ( tokenizion and remove punctuation,remove stop words,lemmatization words,stemming words) manually**"""

text = d1.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc1 = lemmatized_word
file1 = " ".join(doc1)

text = d2.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc2 = lemmatized_word
file2 = " ".join(doc2)

text = d3.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc3 = lemmatized_word
file3 = " ".join(doc3)

text = d4.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc4 = lemmatized_word
file4 = " ".join(doc4)

text = d5.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc5 = lemmatized_word
file5 = " ".join(doc5)

text = d6.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc6 = lemmatized_word
file6 = " ".join(doc6)

text = d7.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc7 = lemmatized_word
file7 = " ".join(doc7)

text = d8.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc8 = lemmatized_word
file8 = " ".join(doc8)

text = d9.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc9 = lemmatized_word
file9 = " ".join(doc9)

text = d10.read()

lower =text.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words

doc10 = lemmatized_word
file10 = " ".join(doc10)

print(file1)

"""# **write the output from all files in the new files**"""

with open('/content/drive/MyDrive/dataOut/file1.txt','w')as f:
  f.write(file1)

with open('/content/drive/MyDrive/dataOut/file2.txt','w')as f:
  f.write(file2)

with open('/content/drive/MyDrive/dataOut/file3.txt','w')as f:
  f.write(file3)

with open('/content/drive/MyDrive/dataOut/file4.txt','w')as f:
  f.write(file4)

with open('/content/drive/MyDrive/dataOut/file5.txt','w')as f:
  f.write(file5)

with open('/content/drive/MyDrive/dataOut/file6.txt','w')as f:
  f.write(file6)

with open('/content/drive/MyDrive/dataOut/file7.txt','w')as f:
  f.write(file7)

with open('/content/drive/MyDrive/dataOut/file8.txt','w')as f:
  f.write(file8)

with open('/content/drive/MyDrive/dataOut/file9.txt','w')as f:
  f.write(file9)

with open('/content/drive/MyDrive/dataOut/file10.txt','w')as f:
  f.write(file10)

"""# **read the new files**"""

out1 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file1.txt','r').read())
out2 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file2.txt','r').read())
out3 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file3.txt','r').read())
out4 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file4.txt','r').read())
out5 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file5.txt','r').read())
out6 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file6.txt','r').read())
out7 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file7.txt','r').read())
out8 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file8.txt','r').read())
out9 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file9.txt','r').read())
out10 = tokenizer.tokenize(open('/content/drive/MyDrive/dataOut/file10.txt','r').read())

"""# **Display the most frequent 4 words in ecah document**"""

from nltk.probability import FreqDist
freq = FreqDist(out1) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out2) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out3) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out4) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out5) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out6) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out7) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out8) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out9) #freq for each word in decument
print(freq.most_common(4))
freq = FreqDist(out10) #freq for each word in decument
print(freq.most_common(4))

"""# **create bag of words called (wordset) from all files 'union' & will be used to create our dictionary, then use the dictionary to calculate tf,idf,tf-idf for all files**"""

wordSet = set(out1).union(set(out2)).union(set(out3)).union(set(out4)).union(set(out5)).union(set(out6)).union(set(out7)).union(set(out8)).union(set(out9)).union(set(out10))

wordDictdoc1 = dict.fromkeys(wordSet, 0)
wordDictdoc2 = dict.fromkeys(wordSet, 0)
wordDictdoc3 = dict.fromkeys(wordSet, 0)
wordDictdoc4 = dict.fromkeys(wordSet, 0)
wordDictdoc5 = dict.fromkeys(wordSet, 0)
wordDictdoc6 = dict.fromkeys(wordSet, 0)
wordDictdoc7 = dict.fromkeys(wordSet, 0)
wordDictdoc8 = dict.fromkeys(wordSet, 0)
wordDictdoc9 = dict.fromkeys(wordSet, 0)
wordDictdoc10 = dict.fromkeys(wordSet, 0)

"""# **compute the numper of occurrence for each word of each decument**"""

for word in out1:
    wordDictdoc1[word]+=1

for word in out2:
    wordDictdoc2[word]+=1

for word in out3:
    wordDictdoc3[word]+=1

for word in out4:
    wordDictdoc4[word]+=1

for word in out5:
    wordDictdoc5[word]+=1

for word in out6:
    wordDictdoc6[word]+=1

for word in out7:
    wordDictdoc7[word]+=1

for word in out8:
    wordDictdoc8[word]+=1

for word in out9:
    wordDictdoc9[word]+=1

for word in out10:
    wordDictdoc10[word]+=1


pd.DataFrame([wordDictdoc1, wordDictdoc2, wordDictdoc3, wordDictdoc4, wordDictdoc5, wordDictdoc6, wordDictdoc7, wordDictdoc8, wordDictdoc9, wordDictdoc10])

"""# **compute TF for all decuments**"""

def computeTF(wordDict, bow):
    tfDict = {}
    bowCount = len(bow)
    for word, count in wordDict.items():
        tfDict[word] = count/float(bowCount)
    return tfDict

tfdoc1 = computeTF(wordDictdoc1, out1)
tfdoc2 = computeTF(wordDictdoc2, out2)
tfdoc3 = computeTF(wordDictdoc3, out3)
tfdoc4 = computeTF(wordDictdoc4, out4)
tfdoc5 = computeTF(wordDictdoc5, out5)
tfdoc6 = computeTF(wordDictdoc6, out6)
tfdoc7 = computeTF(wordDictdoc7, out7)
tfdoc8 = computeTF(wordDictdoc8, out8)
tfdoc9 = computeTF(wordDictdoc9, out9)
tfdoc10 = computeTF(wordDictdoc10, out10)

tf_doc =  [tfdoc1,tfdoc2,tfdoc3,tfdoc4,tfdoc5,tfdoc6,tfdoc7,tfdoc8,tfdoc9,tfdoc10]
pd.DataFrame(tf_doc)

"""# **compute IDF for all decuments**"""

def computeIDF(docList):
    import math
    idfDict = {}
    N = len(docList)

    idfDict = dict.fromkeys(docList[0].keys(), 0)
    for doc in docList:
        for word, val in doc.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log10(N / float(val))

    return idfDict

idfs = computeIDF([wordDictdoc1, wordDictdoc2, wordDictdoc3, wordDictdoc4, wordDictdoc5, wordDictdoc6, wordDictdoc7, wordDictdoc8, wordDictdoc9, wordDictdoc10])
pd.DataFrame([idfs])

"""# **compute TF-IDF for all decuments**"""

def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = val*idfs[word]
    return tfidf

tfidfdoc1 = computeTFIDF(tfdoc1, idfs)
tfidfdoc2 = computeTFIDF(tfdoc2, idfs)
tfidfdoc3 = computeTFIDF(tfdoc3, idfs)
tfidfdoc4 = computeTFIDF(tfdoc4, idfs)
tfidfdoc5 = computeTFIDF(tfdoc5, idfs)
tfidfdoc6 = computeTFIDF(tfdoc6, idfs)
tfidfdoc7 = computeTFIDF(tfdoc7, idfs)
tfidfdoc8 = computeTFIDF(tfdoc8, idfs)
tfidfdoc9 = computeTFIDF(tfdoc9, idfs)
tfidfdoc10 = computeTFIDF(tfdoc10, idfs)

pd.DataFrame([tfidfdoc1,tfidfdoc2,tfidfdoc3,tfidfdoc4,tfidfdoc5,tfidfdoc6,tfidfdoc7,tfidfdoc8,tfidfdoc9,tfidfdoc10])

"""# **The Quary with applying preprocessing on it**"""

#Quary
Quary = "what is the best diet for healthy lifestyle "

lower =Quary.lower()  #lower casing
tokenizer = nltk.RegexpTokenizer(r"\w+")
new_words = tokenizer.tokenize(lower)   #tokenizion and remove punctuation

stopword = stopwords.words('english')
removing_stopwords = [word for word in new_words if word not in stopword] #remove stop words

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized_word = [wordnet_lemmatizer.lemmatize(word) for word in removing_stopwords] #lemmatization words

wordnet_stemming = PorterStemmer()
stemming_word = [wordnet_stemming.stem(word) for word in lemmatized_word] #stemming words


final = lemmatized_word

"""# **compute TF-IDF for all decuments for the query**"""

# tf-idf score across all docs for the query string("what diet for healthy lifestyle")
def compute_tfidf_with_alldocs(documents , query):
    tf_idf = []
    index = 0
    query_tokens = query
    df = pd.DataFrame(columns=['doc'] + query_tokens)
    for doc in documents:
        df['doc'] = np.arange(0 , len(documents))
        doc_num = tf_doc[index]
        sentence = doc
        for word in sentence:
            for text in query_tokens:
                if(text == word):
                    idx = sentence.index(word)
                    tf_idf_score = doc_num[word] * idfs[word]
                    tf_idf.append(tf_idf_score)
                    df.iloc[index, df.columns.get_loc(word)] = tf_idf_score
        index += 1
    df.fillna(0 , axis=1, inplace=True)
    return tf_idf , df

documents = [out1, out2, out3, out4 , out5, out6, out7, out8, out9, out10]
tf_idf , df = compute_tfidf_with_alldocs(documents , final)
print(df)

"""# **create dictionary for the query to calculate the number of each word occurrence in the query**"""

wordSetQ = set(final)
DictQ = dict.fromkeys(wordSetQ, 0)

"""# **compute the number of occurrence to each word in the query**"""

for word in final:
    DictQ[word]+=1
pd.DataFrame([DictQ])

"""# **compute TF & IDF for Query**"""

#Normalized Term Frequency
def termFrequency(term, document):
    normalizeDocument = document
    return normalizeDocument.count(term) / float(len(normalizeDocument))




def inverseDocumentFrequency(term, allDocuments):
    numDocumentsWithThisTerm = 0
    for doc in range (0, len(allDocuments)):
        if term in allDocuments[doc]:
            numDocumentsWithThisTerm = numDocumentsWithThisTerm + 1

    if numDocumentsWithThisTerm > 0:
        return math.log10(float(len(allDocuments)) / numDocumentsWithThisTerm)
    else:
        return 1.0

#Normalized TF for the query string("what diet for healthy lifestyle")
def compute_query_tf(query):
    query_norm_tf = {}
    tokens = query
    for word in tokens:
        query_norm_tf[word] = termFrequency(word , query)
    return query_norm_tf
query_norm_tf = compute_query_tf(final)
pd.DataFrame([query_norm_tf])

"""# **compute IDF for Query**"""

#idf score for the query string("what diet for healthy lifestyle")
def compute_query_idf(query):
    idf_dict_qry = {}
    sentence = query
    documents = [out1, out2, out3, out4, out5, out6, out7, out8, out9, out10]
    for word in sentence:
        idf_dict_qry[word] = inverseDocumentFrequency(word ,documents)
    return idf_dict_qry
idf_dict_qry = compute_query_idf(final)
pd.DataFrame([idf_dict_qry])

"""# **compute TF-IDF for Query**"""

#tf-idf score for the query string("what diet for healthy lifestyle")
def compute_query_tfidf(query):
    tfidf_dict_qry = {}
    sentence = query
    for word in sentence:
        tfidf_dict_qry[word] = query_norm_tf[word] * idf_dict_qry[word]
    return tfidf_dict_qry
tfidf_dict_qry = compute_query_tfidf(final)
pd.DataFrame([tfidf_dict_qry])

"""# **calculate the similarity between query and all documents, Use (Cosine Similarity)**"""

#Cosine Similarity(Query,Document1) = Dot product(Query, Document1) / ||Query|| * ||Document1||

def cosine_similarity(tfidf_dict_qry, df , query , doc_num):
    dot_product = 0
    qry_mod = 0
    doc_mod = 0
    tokens = query

    for keyword in tokens:
        dot_product += tfidf_dict_qry[keyword] * df[keyword][df['doc'] == doc_num]
        #||Query||
        qry_mod += tfidf_dict_qry[keyword] * tfidf_dict_qry[keyword]
        #||Document||
        doc_mod += df[keyword][df['doc'] == doc_num] * df[keyword][df['doc'] == doc_num]
    qry_mod = np.sqrt(qry_mod)
    doc_mod = np.sqrt(doc_mod)
    #implement formula
    denominator = qry_mod * doc_mod
    cos_sim = dot_product/denominator

    return cos_sim

from collections import Iterable
def flatten(lis):
     for item in lis:
        if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                yield x
        else:
             yield item

"""# **Ranking the result of similarity for each document and order similarity values from the upper value " close to 1 " to the lower " close to zero "**"""

def rank_similarity_docs(data):
    cos_sim =[]
    for doc_num in range(0 , len(data)):
        cos_sim.append(cosine_similarity(tfidf_dict_qry, df , final , doc_num))
    return cos_sim
files = [out1,out2,out3,out4,out5,out6,out7,out8,out9,out10]
similarity_docs = rank_similarity_docs(files)

res_sim = list(flatten(similarity_docs))
res_sim.sort(reverse=True)

f = pd.DataFrame({
    "Cosine similarty": res_sim,
    "NO.Decument": [1,2,3,4,5,6,7,8,9,10]
})
f
from natsort import index_natsorted
f.sort_values(
    by = "Cosine similarty",
    ascending= False,
    key = lambda data: np.argsort(index_natsorted(f["Cosine similarty"]))
)

"""# **Display the relevant and non-relevant**"""

i = 1
for x in res_sim:
 if math.isnan(x):
    print("document",(i),":  irrelevant (No Result)")
    i = i+1
 else:
      print("document",(i),":  relevant")
      i = i+1