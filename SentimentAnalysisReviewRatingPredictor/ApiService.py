import re
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize


def train_classifier(train_reviews, stop, lm, words):
    train_reviews = remove_empty_columns(train_reviews)
    train_reviews = prepare_review_data(train_reviews, stop, lm, words)
    classifier_train_data = create_classifier_data(train_reviews)
    # You can change the classifier type by modifying below
    cl = nltk.ConditionalExponentialClassifier.train(classifier_train_data, max_iter=2)
    return cl


def prepare_test_data(test_reviews, stop, lm, words):
    test_reviews = remove_empty_columns(test_reviews)
    test_reviews = prepare_review_data(test_reviews, stop, lm, words)
    classifier_test_data = create_classifier_data(test_reviews)
    return classifier_test_data


def prepare_review_data(data, stop, lm, words):
    data['text'] = data['text'].astype(str)
    data['text'] = data['text'].apply(lambda x: " ".join(x.lower() for x in x.split()))
    data['text'] = data['text'].str.replace('[^\w\s]|\d', '')
    data['text'] = data['text'].apply(lambda x: " ".join(x for x in x.split() if x in words and x not in stop))
    data['text'] = data['text'].apply(lambda x: lemmatize_sent(x, lm))
    return data


def create_word_features(words):
    my_dict = dict([(word, True) for word in word_tokenize(words)])
    return my_dict


def remove_empty_columns(data):
    data = data.dropna(how='all', axis=1)
    data = data[0:]
    return data


def create_classifier_data(data):
    pos_reviews = []
    neg_reviews = []
    s_neg_reviews = []
    s_pos_reviews = []
    neu_reviews = []
    for index, row in data.iterrows():
        my_dict = dict([(word, True) for word in word_tokenize(row['text'])])
        if row['rating'] == 5:
            pos_reviews.append((my_dict, "pos"))
        elif row['rating'] == 4:
            s_pos_reviews.append((my_dict, "s-pos"))
        elif row['rating'] == 3:
            neu_reviews.append((my_dict, "neu"))
        elif row['rating'] == 2:
            s_neg_reviews.append((my_dict, "s-neg"))
        else:
            neg_reviews.append((my_dict, "neg"))
    classifier_data = pos_reviews + s_pos_reviews + neu_reviews + s_neg_reviews + neg_reviews
    return classifier_data


def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag, wordnet.NOUN)


def penn2morphy(penntag):
    morphy_tag = {'NN': 'n', 'JJ': 'a',
                  'VB': 'v', 'RB': 'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n'


def lemmatize_sent(text, lm):
    result = ''
    for word, tag in pos_tag(word_tokenize(text)):
        if tag != 'NN':
            lem = lm.lemmatize(word, pos=penn2morphy(tag))
            result = result + " " + lem
    return result


def process_review(review, stop, lm, cl, words):
    review = review.lower()
    review = re.sub('[^\w\s]|\d', '', review)
    review = ' '.join([word for word in review.split() if word in words and word not in stop])
    review = lemmatize_sent(review, lm)
    result = cl.prob_classify(create_word_features(review))
    result = cl.classify(create_word_features(review))
    return result
