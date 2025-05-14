import requests
from textblob import TextBlob
import pandas as pd
from newspaper import Article

url = 'https://en.wikipedia.org/wiki/OpenAI'
article = Article(url)
article.download()
article.parse()
text = article.text
text_blob = TextBlob(text)
polarity = text_blob.polarity
subjectivity = text_blob.subjectivity
print(f"Polarity of overall article: {polarity}")
print(f"Subjectivity of overall article: {subjectivity}")

sentences = text.split('\n\n')
sentence_sentiments = []
for sentence in sentences:
    sentence_sentiment = TextBlob(sentence).sentiment.polarity
    sentence_sentiments.append(sentence_sentiment)

print("Sentiment of the text:", polarity)
print("Sentiment of individual sentences:", sentence_sentiments)

lines = text.split('\n')

line_texts = []
polarities = []
subjectivities = []

for line in lines:
    if line:
        line_blob = TextBlob(line)
        polarity = line_blob.sentiment.polarity
        subjectivity = line_blob.sentiment.subjectivity
        
        line_texts.append(line)
        polarities.append(polarity)
        subjectivities.append(subjectivity)

data = {'Line Text': line_texts, 'Polarity': polarities, 'Subjectivity': subjectivities}
df = pd.DataFrame(data)
print(df)

sorted_by_polarity = df.sort_values('Polarity', ascending=False)
sorted_by_subjectivity = df.sort_values('Subjectivity', ascending=False)
print("5 Most Positive Sentences:")
for i, row in sorted_by_polarity.head(5).iterrows():
    print(f"{row['Line Text']} (p = {row['Polarity']:.3f}, s = {row['Subjectivity']:.3f})")

print("\n5 Most Negative Sentences:")
for i, row in sorted_by_polarity.tail(5).iterrows():
    print(f"{row['Line Text']} (p = {row['Polarity']:.3f}, s = {row['Subjectivity']:.3f})")

print("\n5 Most Objective Statements:")
for i, row in sorted_by_subjectivity.head(5).iterrows():
    print(f"{row['Line Text']} (p = {row['Polarity']:.3f}, s = {row['Subjectivity']:.3f})")

print("\n5 Most Subjective Statements:")
for i, row in sorted_by_subjectivity.tail(5).iterrows():
    print(f"{row['Line Text']} (p = {row['Polarity']:.3f}, s = {row['Subjectivity']:.3f})")

