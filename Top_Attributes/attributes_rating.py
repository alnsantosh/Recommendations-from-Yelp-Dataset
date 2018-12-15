from gensim.test.utils import datapath
from gensim.models import LdaModel
import json
from gensim.corpora import Dictionary
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import gensim
import sys
from pycorenlp import StanfordCoreNLP
from random import shuffle


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

    lda_model_tfidf = LdaModel.load(model_file)
    dictionary = Dictionary.load_from_text(dict_file)
    stemmer = SnowballStemmer('english')

    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))


    # test = "They have really good iced tea, avaiable in 3 flavors- including unsweetened peach black tea and passion berry (amazing!)."
    # bow_vector = dictionary.doc2bow(preprocess(test, stemmer))
    # (topic, score) = sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1])[0]
    # print(topic)
    # sys.exit()

    nlp = StanfordCoreNLP('http://localhost:9000')

    review_data = None
    with open('yelp_dataset/cleaned_reviews.json') as file:
        review_data = [json.loads(line) for line in file]

    review_data = review_data[:100000]
    shuffle(review_data)
    review_data = review_data[:4000]

    print("Input Reading Done",len(review_data))
    attributes= ["Food","Service","Value For Money","Ambience"]

    topics = [None] * 10
    topics[0]= "Service"
    topics[1] = "Ambience"
    topics[2] = "Value For Money"
    topics[3] = "Service"
    topics[4] = "Food"
    topics[5] = "Service"
    topics[6] = "Ambience"
    topics[7] = "Food"
    topics[8] = "Value For Money"
    topics[9] = "Food"

    restaurants_attributes = {}
    count = 0
    restaurants_stars = {}

    for i in range(len(review_data)):
        if count%1000==0:
            print(count)
        count+=1
        business_id = review_data[i]["business_id"]
        if business_id not in restaurants_attributes:
            restaurants_stars[business_id]=(0,0)
            restaurants_attributes[business_id]={}
            for attr in attributes:
                restaurants_attributes[business_id][attr] = (0, 0)
        stars,ct = restaurants_stars[business_id]
        ct+=1
        stars+=(review_data[i]["stars"]-1)
        restaurants_stars[business_id] = (stars,ct)
        sentence = review_data[i]['text']
        sentence = sentence.split('.')
        for i in sentence:
            if len(i) > 0:
                bow_vector = dictionary.doc2bow(preprocess(i, stemmer))
                (topic,score) = sorted(lda_model_tfidf[bow_vector], key=lambda tup: -1 * tup[1])[0]
                attribute = topics[topic]
                if attribute is not None:
                    res = nlp.annotate(i,properties={
                                           'annotators': 'sentiment',
                                           'outputFormat': 'json',
                                           'timeout': 1000,
                                       })
                    if res and "sentences" in res and len(res["sentences"])>0:
                        # print(i,res)
                        temp = res["sentences"][0]
                        sentimentValue = int(temp["sentimentValue"])
                        # sys.exit()
                        (total,count) = restaurants_attributes[business_id][attribute]
                        total+=sentimentValue
                        count+=1
                        restaurants_attributes[business_id][attribute] = (total, count)
    del review_data

    f = open("results.txt", "a")
    for rest in restaurants_attributes:
        f.write(str(rest)+"\t")
        stars, ct = restaurants_stars[rest]
        avg_stars = str(round(stars / ct, 2))
        f.write("Stars : "+avg_stars + "\t")
        for attr in attributes:
            (total,count) = restaurants_attributes[rest][attr]
            if count>0:
                avg_rating = str(round(total/count, 2))

                f.write(str(attr)+" : "+avg_rating+"\t")
        f.write("\n")
    f.close()