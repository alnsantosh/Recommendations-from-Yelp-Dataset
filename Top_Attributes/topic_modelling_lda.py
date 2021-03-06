import json
import sys
import gensim
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim import corpora, models
from gensim.test.utils import datapath
from random import shuffle
from gensim.models import LdaModel


def lemmatize_stemming(text,stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text,stemmer):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token,stemmer))
    return result


if __name__ == '__main__':
    review_data = None
    with open('yelp_dataset/cleaned_reviews.json') as file:
        review_data = [json.loads(line) for line in file]
    docs = []
    for i in range(len(review_data)):
        sentence = review_data[i]['text']
        sentence = sentence.split('.')
        for i in sentence:
            if len(i) > 0:
                docs.append(i)
    print("Input Reading Done")
    shuffle(docs)
    docs = docs[:10000000]

    print(len(review_data), len(docs))

    del review_data
    processed_docs = []
    stemmer = SnowballStemmer('english')


    for i in range(len(docs)):
        if i%10000==0:
            print(i)
        processed_docs.append(preprocess(docs[i],stemmer))
    print("Pre-processing Done")

    del docs

    dictionary = gensim.corpora.Dictionary(processed_docs)
    dictionary.filter_extremes(no_below=1000, no_above=0.5, keep_n=100000)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    del processed_docs

    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus]

    print("Entering TFIDF")


    #TFIDF
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=5, id2word=dictionary, passes=2, workers=4)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))

    # temp_file = datapath("model")
    dictionary.save_as_text("model_new_dict")
    lda_model_tfidf.save("Models/model_new")

