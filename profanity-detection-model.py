#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install --upgrade python_version
import pandas as pd
#!pip install seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import re
#!pip install pyforest
from pyforest import *


# In[2]:


#!pip install nltk
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


# In[3]:


## Installing Profanity Check
#!pip install profanity-check

## Installing PyTorch
#!pip install torch


# In[4]:


from profanity_check import predict, predict_prob

predict(['predict() takes an array and returns a 1 for each string if it is offensive, else 0.'])
# [0]

predict(['fuck off'])
# [1]


# #### Inference: 1 means it's a profane word.

# In[5]:


predict_prob(['predict_prob() takes an array and returns the probability each string is offensive'])
# [0.08686173]

predict_prob(['fuck off'])


# #### The confidence interval of the word bastard for being profane is 0.85

# ## Reading file with Pandas 

# In[6]:


label_data= pd.read_csv('labeled_data.csv',encoding='iso-8859-1')
label_data.head()


# #### Already we have index for the row data. Hence dropping the column, Unnamed: 0

# In[7]:


label_data.drop(columns=["Unnamed: 0"], axis=1, inplace= True)
label_data.head()


# In[8]:


label_data.isnull().sum()


# In[9]:


label_data= label_data.dropna()
label_data.shape


# In[10]:


label_data.dtypes


# In[11]:


label_data["count"]= label_data["count"].astype('int64')
label_data["hate_speech"]= label_data["hate_speech"].astype('int64')
label_data["offensive_language"]= label_data["offensive_language"].astype('int64')
label_data["neither"]= label_data["neither"].astype('int64')
label_data["class"]= label_data["class"].astype('int64')
label_data.head()


# ## Data Description
# 
#     ➡count- number of CrowdFlower users who coded each tweet (min is 3, sometimes more users coded a tweet when judgments 
#     were determined to be unreliable by CF).
# 
#     ➡hate_speech- number of CF users who judged the tweet to be hate speech.
# 
#     ➡offensive_language- number of CF users who judged the tweet to be offensive.
# 
#     ➡neither- number of CF users who judged the tweet to be neither offensive nor non-offensive.
# 
#     ➡class- class label for majority of CF users. [0 - hate speech, 1 - offensive language, 2 - neither]
#     
#     ➡tweet- message posted by CF users

# In[12]:


sns.countplot(label_data["hate_speech"])


# In[13]:


sns.countplot(label_data["offensive_language"])


# In[14]:


sns.countplot(label_data["neither"])


# In[15]:


label_data["class"].value_counts()


# ### Visualizing Class 

# In[16]:


n_class = [56, 762, 182]
labels= "hate_speech", "offensive_language", "neither"
colors= ["blue", "red","grey"]
explode=[0,0.1,0.1]
plt.figure(figsize=(6,6))
plt.pie(n_class, labels=labels, colors=colors, explode=explode, shadow=True, autopct= "%.2f%%")
plt.title("Class (Pie Chart)", fontsize=15)
plt.axis("off")
plt.show()


# # Generative Pre-trained Transformer (GPT-2)

# In[17]:


#!pip install transformers==3.1.0


# In[18]:


from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')


# In[19]:


label_data.tweet.iloc[0]


# In[20]:


# Allocate a pipeline for sentiment-analysis
def sentiment_pred(text):
    classifier = pipeline('sentiment-analysis')
    return classifier(text)


# In[21]:


sentiment_pred('!!! RT @mayasolovely: As a woman you shouldnt complain about cleaning up your house. &amp; as a man you should always take the trash ouï¿½')


# In[22]:


### Data cleaning/Preparation 

def clean_text(text):
    text = text.lower()
    text = re.sub(r"what's", "what is ", text)  ## conversion of contraction words to expanded words
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text= re.sub(r"couldn't","could not",text)
    text= re.sub(r"shouldn't","should not",text)
    text= re.sub(r"wouldn't","would not",text)
    text= re.sub(r"shalln't","shall not",text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    text = re.sub('\W', ' ', text)              ## removing non-word characters
    text = re.sub('[^A-Za-z\' ]+', '',text)     ## removing all non-alphanumeric values(Except single quotes)
    text = re.sub('\s+', ' ', text)
    text = text.strip(' ')
    text = ' '.join([word for word in text.split() if word not in (stop_words)])    ### Stopwords removal
    return text

label_data["tweet"] = label_data["tweet"].apply(clean_text)


# ## Tweets Sentiments Prediction

# In[23]:


classifier = pipeline('sentiment-analysis')
cust_tweets= list(label_data.tweet.iloc[0:1000])
sentiments_preds= classifier(cust_tweets)
sentiments_preds


# In[24]:


len(sentiments_preds)


# In[25]:


label_data.head()


# In[26]:


classifier_sentiments= pd.DataFrame(sentiments_preds)
count= list(label_data["count"])
hate_speech= list(label_data["hate_speech"])
offense= list(label_data["offensive_language"])
neither= list(label_data["neither"])
class_1= list(label_data["class"])
tweets= list(label_data["tweet"])
classifier_sentiments["count"]= count
classifier_sentiments["hate_speech"]= hate_speech
classifier_sentiments["offensive_language"]= offense
classifier_sentiments["neither"]= neither
classifier_sentiments["class"]= class_1
classifier_sentiments["tweet"]= tweets
classifier_sentiments.rename(columns={"score": "confidence_interval","label":"sentiment_label"}, inplace=True)
colos= ["count", "hate_speech","offensive_language","neither","class","tweet", "sentiment_label", "confidence_interval"]
classifier_sentiments= classifier_sentiments[colos]
classifier_sentiments.head()


# In[27]:


classifier_sentiments.sentiment_label.value_counts()


# ## Visualizing Profanity for Sentiment Analysis

# In[28]:


profane_class = [120, 880]
labels= "POSITIVE", "NEGATIVE"
colors= ["green", "red"]
explode=[0,0.1]
plt.figure(figsize=(6,6))
plt.pie(profane_class, labels=labels, colors=colors, explode=explode, shadow=True, autopct= "%.2f%%")
plt.title("Profanity Classication", fontsize=30)
plt.axis("off")
plt.show()


# ## Pickling the model

# In[30]:


#pickling the model
import joblib as jl
jl.dump(classifier, "profanity-detection-model.pkl")
msg= 'Who the bloody hell are you fuckkard?'
from_jb= jl.load("profanity-detection-model.pkl")
from_jb.predict([msg])

