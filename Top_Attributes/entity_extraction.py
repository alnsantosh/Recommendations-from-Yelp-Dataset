import json




if __name__ == '__main__':
    review_data = None

    with open('yelp_dataset/yelp_academic_dataset_review.json') as f:
        review_data = [json.loads(line) for line in f]

    print(review_data[0],len(review_data))
