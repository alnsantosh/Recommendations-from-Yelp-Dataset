import json
import nltk


if __name__ == '__main__':
    review_data = None
    with open('yelp_dataset/cleaned_reviews.json') as file:
        review_data = [json.loads(line) for line in file]
    output = []
    count = 0
    for i in review_data:
        if i['useful']>0:
            count+=1
    print(count,len(review_data))
    # print(len(review_data))
    # for i in range(10):
    #     sentence = review_data[i]['text']
    #     tokens = nltk.word_tokenize(sentence)
    #     tagged = nltk.pos_tag(tokens)
    #     entities = nltk.chunk.ne_chunk(tagged)
    #     output.append(entities)
    # for i in output:
    #     print(i)
