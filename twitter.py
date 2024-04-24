import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Exp3_twitter-sentiment-analysis.csv')
df.drop(columns=df.columns[df.columns != 'tweet'], inplace=True)
df

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))


def preprocess(t):
    w = word_tokenize(t.lower())
    data = list()
    for word in w:
        if word.lower() not in stop_words:
            data.append(word)
    return ' '.join(data)
df['tweet'] = df['tweet'].apply(preprocess)
df['tweet'].reset_index(drop=True, inplace=True)

import re
def remove_special_characters(text):
    pattern = r'[^a-zA-Z0-9\s]'
    clean_text = re.sub(pattern, '', text)
    return clean_text
df['tweet'] = df['tweet'].apply(remove_special_characters)

import string
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
df['tweet'] = df['tweet'].apply(remove_punctuation)
df['tweet'].reset_index(drop=True, inplace=True)

df.isnull().sum()

from textblob import TextBlob
def get_subjectivity(text):
    return TextBlob(text).sentiment.subjectivity
def get_polarity(text):
    return TextBlob(text).sentiment.polarity
df['subjectivity'] = df['tweet'].apply(get_subjectivity)
df['polarity'] = df['tweet'].apply(get_polarity)

df

threshold = 0.05
df['sentiment'] = df['polarity'].apply(lambda x: ('Positive' if x
        >= threshold else ('Negative' if x < -threshold else 'Neutral'
        )))
df['sentiment'].value_counts()

df['sentiment'].value_counts().plot.barh()

df_positive = df[df['sentiment'] == 'Positive']
df_negative = df[df['sentiment'] == 'Negative']
df_neutral = df[df['sentiment'] == 'Neutral']

from wordcloud import WordCloud
wordcloud = WordCloud(width=800, height=400, background_color='white'
                    ).generate(' '.join(df_positive['tweet']))
plt.figure(figsize=(10, 5))
plt.title('WordCloud for positive sentiments')
plt.imshow(wordcloud, interpolation='lanczos')
plt.axis('off')
plt.show()

wordcloud = WordCloud(width=800, height=400, background_color='white'
                    ).generate(' '.join(df_negative['tweet']))
plt.figure(figsize=(10, 5))
plt.title('WordCloud for negatvie sentiments')
plt.imshow(wordcloud, interpolation='lanczos')
plt.axis('off')
plt.show()
