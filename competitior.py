import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))

import string

def preprocess(t):
    w = word_tokenize(t.lower())
    data = list()
    for word in w:
        if word.lower() not in stop_words:
            data.append(word)
    text = ' '.join(data)
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


import re
def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    clean_text = re.sub(pattern, '', text)
    return clean_text


from textblob import TextBlob


def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity


def get_polarity(text):
    return TextBlob(text).sentiment.polarity


df = pd.read_csv('k_competitor.csv')
df

apple = pd.DataFrame({'Review': df['apple']})
apple.head()

garbage = apple['Review'].apply(lambda x: isinstance(x, float))
apple = apple[~garbage]

apple['Review'] = apple['Review'].apply(preprocess)
apple['Review'] = apple['Review'].apply(remove_special_characters)
apple['Review'] = apple['Review'].apply(remove_punctuation)
apple['Review'].reset_index(drop=True, inplace=True)
apple['subjectivity'] = apple['Review'].apply(get_subjectivity)
apple['polarity'] = apple['Review'].apply(get_polarity)
apple.head()

threshold = 0.05
apple['sentiment'] = apple['polarity'].apply(lambda x: ('Positive' if x
        >= threshold else ('Negative' if x < -threshold else 'Neutral'
        )))

sentiment_counts = apple['sentiment'].value_counts()
plt.barh(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Count')
plt.ylabel('Sentiment')
plt.title('Sentiment Distribution in Apple Data')
plt.show()

apple_sentiments = apple.groupby('sentiment')['sentiment'].count()
print("Apple",apple_sentiments)

samsung = pd.DataFrame({'Review': df['samsung']})
samsung.head()

garbage = samsung['Review'].apply(lambda x: isinstance(x, float))
samsung = samsung[~garbage]

samsung['Review'] = samsung['Review'].apply(preprocess)
samsung['Review'] = samsung['Review'].apply(remove_special_characters)
samsung['Review'] = samsung['Review'].apply(remove_punctuation)
samsung['Review'].reset_index(drop=True, inplace=True)
samsung['subjectivity'] = samsung['Review'].apply(get_subjectivity)
samsung['polarity'] = samsung['Review'].apply(get_polarity)
samsung.head()

threshold = 0.05
samsung['sentiment'] = samsung['polarity'].apply(lambda x: ('Positive'
        if x >= threshold else ('Negative' if x
        < -threshold else 'Neutral')))

sentiment_counts = samsung['sentiment'].value_counts()
plt.barh(sentiment_counts.index, sentiment_counts.values)
plt.xlabel('Count')
plt.ylabel('Sentiment')
plt.title('Sentiment Distribution of Samsung Data')
plt.show()

samsung_sentiments = samsung.groupby('sentiment')['sentiment'].count()
print("Samsung",samsung_sentiments)

colors = ['red', 'blue']
(fig, ax) = plt.subplots()
apple_sentiments.plot(
    kind='bar',
    ax=ax,
    position=0,
    width=0.4,
    label='apple',
    color=colors[0],
    )
samsung_sentiments.plot(
    kind='bar',
    ax=ax,
    position=1,
    width=0.4,
    label='samsung',
    color=colors[1],
    )
ax.set_xlabel('Sentiment Label')
ax.set_ylabel('Number of Tweets')
ax.legend()
plt.show()
