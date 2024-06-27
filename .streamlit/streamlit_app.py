import streamlit as st
st.set_page_config(page_title='EduSentiment', page_icon='🏫')

import pandas as pd
import pickle
import re
from googletrans import Translator, constants
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer, PorterStemmer
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier

import nltk
import spacy
import string
import warnings
import numpy as np
import seaborn as sns 
import joblib
import unicodedata
from collections import Counter
import matplotlib.pyplot as plt 
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer, word_tokenize
from wordcloud import WordCloud, STOPWORDS
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

@st.cache_data
def load_model_and_vectorizer():
    # Load model and vectorizer
    with open('rf_model7.pkl', 'rb') as file:
        model = pickle.load(file)           
    with open('tfidf_vect_chars.pkl', 'rb') as file:
        vectorizer = pickle.load(file)
    return model, vectorizer

# Load model and vectorizer
model, vectorizer = load_model_and_vectorizer()

# Initialize the translator
translator = Translator()

@st.cache_data
# Function to translate text from Malay to English
def translate_text(text):
    # Translate the text
    translation = translator.translate(text, src='ms', dest='en')

    # Check if the translation is None
    if translation is None:
        # If the translation is None, return the original text
        return text
    else:
        # Otherwise, return the translated text
        return translation.text

@st.cache_data
# Define a function to map POS tags to WordNet POS tags
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN
    
@st.cache_data
# Function for text preprocessing
def preprocess_text(text):

    # Initialize WordNet Lemmatizer and stopwords
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # Convert text to lowercase
    text = text.lower()

    # Normalize text (Remove accents, diacritics, etc.)
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    
    # Remove special characters (except for words starting with 'not_')
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Initialize a variable to keep track of whether the current token follows a negation
    following_negation = False
    
    # Iterate over tokens and modify sentiment of words that follow negations
    for i in range(len(tokens)):
        token = tokens[i]
        if token in ["not", "no"]:
            following_negation = True
        elif following_negation:
            # Modify sentiment of the word
            tokens[i] = "not_" + tokens[i]
            following_negation = False
    
    # Remove punctuation
    tokens = [token for token in tokens if token not in string.punctuation]
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]

    # Remove non-alphanumeric characters except for words starting with 'not_'
    tokens = [token for token in tokens if token.isalnum() or token.startswith('not_')]
    
    # Whitespace removal
    tokens = [token.strip() for token in tokens]
    
    # POS tags for lemmatization
    pos_tags = pos_tag(tokens)

    # Lemmatization
    tokens = [lemmatizer.lemmatize(token, get_wordnet_pos(tag)) for token, tag in pos_tags]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text
    

# Page title
st.title('EduSentiment 🏫📚')

with st.expander('About this app'):
  st.markdown('**What can this app do?**')
  st.info('This app allow users like the university administration to analyse predicted sentiments of Course and Teaching Evaluation (CTES) student reviews. The application is built with a machine learning (ML) model, including machine translation and data pre-processing. A sentiment analysis report will be generated to display the distribution of positive and negative sentiments for each subject.')

  st.markdown('**How to use the app?**')
  st.info("""
  To engage with the app, go to the sidebar:
  1. Select and upload a dataset
  2. Wait for the selected dataset to be processed for sentiment prediction using Machine Learning
  3. View the results for sentiment distributions
  4. Download the model results of sentiment prediction and distribution
  """)

  st.markdown('**Under the hood**')
  st.markdown('Data sets:')
  st.code('''- CTES Sem 1 23/24 data set
  ''', language='markdown')
  
  st.markdown('**Contact support**')
  st.info("""
  If you encounter any issues, please contact support at u2005364@siswa.um.edu.my.
  Disclaimer: This prediction model aims to predict the probability of student review sentiment.
  For: UM FSKTM Administration | Powered by: FSKTM FYP WIA3003 23/24 Fu Yik Lyn
  """)


# Sidebar
with st.sidebar:
    # Load data
    st.header('Input data')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, index_col=False, encoding='iso-8859-1')
    elif uploaded_file is None:
        st.warning("Please upload a CSV file!")
    else:
        st.success('File successfully uploaded and processed.')
        

if uploaded_file:
    try:       
        # Translate text data from Malay to English
        df['Review'] = df['Review'].apply(translate_text)

        # Preprocess text data
        df['Review'] = df['Review'].apply(preprocess_text)

        df['Review'].replace([''], np.nan, inplace=True)
        df['Review'].replace(['none', 'nothing','nil'], np.nan, inplace=True)

        # Drop rows that have empty or null 'Review' after preprocessing
        df.dropna(subset=['Review'], inplace=True)

        # Save the cleaned DataFrame back to a new CSV file
        cleaned_csv_file = "Cleaned_reviews.csv"
        df.to_csv(cleaned_csv_file, index=False)

        # Predict the sentiment of the reviews
        reviews_vectorized = vectorizer.transform(df['Review'])
        predictions = model.predict(reviews_vectorized)
        df['Sentiment'] = predictions

        # Display the preprocessed dataframe
        st.write(df)

        # Filter function to select the course subject
        subjects = df['Subject'].unique()
        selected_subject = st.selectbox('Select a course subject', subjects)
        
        def display_visuals(selected_subject):
            # Filter the dataframe based on the selected subject
            df_filtered = df[df['Subject'] == selected_subject]

            # Count the number of each sentiment
            sentiment_counts = df_filtered['Sentiment'].value_counts()

            # Display the pie chart for percentage of each sentiment
            st.title("Sentiment Prediction Analysis")
            st.write("Sentiment prediction analysis involves assessing the emotional tone or polarity of text data, such as student reviews or comments. It aims to determine whether the sentiment expressed is positive or negative.")
                     
            fig, ax = plt.subplots()
            ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%')
            ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title(f'Sentiment Distribution for {selected_subject} Courses')
            st.pyplot(fig)

            # Display the bar chart for frequency of sentiments
            fig, ax = plt.subplots()
            ax.bar(sentiment_counts.index, sentiment_counts.values, color=['skyblue', 'skyblue'])
            plt.xlabel('Sentiment')
            plt.ylabel('Frequency')
            plt.title(f'Sentiment Frequency for {selected_subject} Courses')
            plt.xticks(rotation=45)  # Rotate x-axis labels for readability
            st.pyplot(fig)

            # Display bar chart for distibution of review length
            df_filtered['review_length'] = df_filtered['Review'].apply(len)
            st.title("Review Length Distribution")
            st.write("The review length distribution refers to the spread or pattern of review lengths within a dataset. It helps analyze how long or short reviews typically are, which can be useful for understanding user engagement or sentiment.")

            # Create a histogram of review lengths
            fig, ax = plt.subplots()
            ax.hist(df_filtered['review_length'], bins=50, color='skyblue')
            plt.xlabel('Length of the Reviews')
            plt.ylabel('Frequency')
            plt.title('Distribution of Review Lengths')
            st.pyplot(fig)

            #Review analysis using Wordcloud
            st.title("Word Cloud")
            st.write("A word cloud is a visual representation of text data, where the size of each word corresponds to its frequency in the given content. It’s often used to highlight prominent terms or themes within a document or dataset. ")

            all_words = ' '.join(df['Review'])
            wordcloud_all = WordCloud(width=800, height=600, random_state=21, max_font_size=120).generate(all_words)
            st.write('Word Cloud for All Reviews')
            st.image(wordcloud_all.to_array())

            #Positive sentiment wordcloud
            pos_words = ' '.join(df['Review'][df['Sentiment'] == 'positive'])
            wordcloud_pos = WordCloud(width=800, height=600, random_state=21, max_font_size=120).generate(pos_words)
            st.write('Word Cloud for Positive Sentiment Reviews')
            st.image(wordcloud_pos.to_array())

            #Negative sentiment wordcloud
            neg_words = ' '.join(df['Review'][df['Sentiment'] == 'negative'])
            wordcloud_neg = WordCloud(width=800, height=600, random_state=21, max_font_size=120).generate(neg_words)
            st.write('Word Cloud for Negative Sentiment Reviews')
            st.image(wordcloud_neg.to_array())

        display_visuals(selected_subject)

    except pd.errors.ParserError:
            st.error("Error: Please upload a valid CSV file.")

else:
    st.warning('👈 Upload a CSV file to get started!')