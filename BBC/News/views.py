from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet,stopwords
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.models import model_from_json
from keras import backend as K
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import string

label_dict={'business':0,'entertainment':1, 'politics':2, 'sport':3, 'tech':4}
inv_label_dict={0:'business',1:'entertainment', 2:'politics', 3:'sport', 4:'tech'}
tfidf = pickle.load(open("tfidf.pickle", 'rb'))

def RemovePunctuation(sentence):
    return sentence.translate(str.maketrans("","", string.punctuation))

def stemming(sentence):
    tknzr = TweetTokenizer()
    line = tknzr.tokenize(sentence)
    stemmer = SnowballStemmer(language='english')
    s=""
    for i in range(len(line)):
        s =  s + " " + stemmer.stem(line[i])
    #print(np.asarray(s).reshape((1,1)))
    return s

def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)

def lemmatizing(sentence):
    tknzr = TweetTokenizer()
    line = tknzr.tokenize(sentence)
    lemmatizer = WordNetLemmatizer()
    s=""
    for l in line:
        s+= " "+ lemmatizer.lemmatize(l, get_wordnet_pos(l))
    #lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in nltk.word_tokenize(sentence)
    return s

def RemoveStopWords(sentence):
    stop_words = set(stopwords.words('english'))
    line = TweetTokenizer().tokenize(sentence) 
    s=""
    for l in line:
        if l not in stop_words: 
            s+= " "+ l
    return s

# Create your views here.


def index(request):
	
	return render(request,'News/index.html',{})



def res(request):
	txt = request.POST['article']
	txt = RemovePunctuation(RemoveStopWords(stemming(txt)))
	txt = [txt]
	txt = tfidf.transform(txt).toarray()
	print(txt.shape)
	model = load_model("Whole_model.h5")
	pred = model.predict(txt)
	K.clear_session()
	print( inv_label_dict[np.argmax(pred)] )
	ans = { 'ans': inv_label_dict[np.argmax(pred)] }

	return render(request,'News/res.html',ans)
