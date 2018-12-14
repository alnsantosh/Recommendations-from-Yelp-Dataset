from gensim.test.utils import datapath
from gensim.models import LdaModel
import json
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
import sys

def lemmatize_stemming(text,stemmer):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text,stemmer):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token,stemmer))
    return result

if __name__ == '__main__':
    model_file = "Models/model_new"
    dict_file =  "model_new_dict"

    # temp_file = datapath(model_file)
    lda_model_tfidf = LdaModel.load(model_file)
    dictionary = Dictionary.load_from_text(dict_file)

    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))
    sys.exit()

    stemmer = SnowballStemmer('english')

    review_data = None
    with open('yelp_dataset/cleaned_reviews.json') as file:
        review_data = [json.loads(line) for line in file]


    for i in range(len(review_data)):
        sentence = review_data[i]['text']
        sentence = sentence.split('.')
        for i in sentence:
            if len(i) > 0:
                bow_vector = dictionary.doc2bow(preprocess(i, stemmer))
                print(preprocess(i,stemmer))
                print(bow_vector)
                sys.exit()

                #     for index, score in sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1]):
                #         print(text)
                #         print("Score: {}\t Topic: {}".format(score, lda_model_tfidf.print_topic(index, 5)))
                #         print("\n")

