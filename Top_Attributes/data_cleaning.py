import json


if __name__ == '__main__':
    review_data = None
    business_data = None

    with open('yelp_dataset/yelp_academic_dataset_review.json') as f:
        review_data = [json.loads(line) for line in f]
    f.close()

    with open('yelp_dataset/yelp_academic_dataset_business.json') as f:
        business_data = [json.loads(line) for line in f]
    f.close()

    business_restaurant=set()
    for i in business_data:
        if i["categories"] and "Restaurants" in i['categories']:
            business_restaurant.add(i["business_id"])

    cleaned_reviews = []
    for review in review_data:
        if review['business_id'] in business_restaurant:
            cleaned_reviews.append(review)
    print("start")
    outfile = open('yelp_dataset/cleaned_reviews.json', 'a')
    for i in range(len(cleaned_reviews)):
        json_data = json.dumps(cleaned_reviews[i])
        outfile.write(json_data+"\n")
    outfile.close()
    print("done")
