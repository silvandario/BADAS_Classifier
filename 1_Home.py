import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import re
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from textblob import TextBlob
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# Function to set the background color
def set_bg_color(hex_color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {hex_color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Function to preprocess input text for sentiment analysis
def preprocess_text_for_sentiment(text):
    lemmatizer = WordNetLemmatizer()
    # Remove all special characters/ numbers
    text = re.sub("[^a-zA-Z]", " ", text)
    # Convert to lowercase
    text = text.lower()
    # Split into words
    text_words = text.split()
    # Lemmatize and remove stopwords
    processed_text = [lemmatizer.lemmatize(word) for word in text_words if word not in set(stopwords.words("english"))]
    # Join words back to string
    return " ".join(processed_text)

# Load the sentiment model
sentiment_model = load_model('nn_sentiment.h5')

# Load the spam model
spam_model = load_model('spam.h5')

# Function for sentiment analysis
def analyze_sentiment(text):
    processed_text = preprocess_text_for_sentiment(text)
    max_features = 10000  # Adjust as per your model
    max_len = 200  # Adjust as per your model
    encoded_input = one_hot(processed_text, n=max_features)
    padded_input = pad_sequences([encoded_input], maxlen=max_len, padding="pre")
    prediction = sentiment_model.predict(padded_input)
    sentiment_label = np.argmax(prediction, axis=1)
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}  # Adjust as per your model
    return sentiment_map.get(sentiment_label[0])

from keras.preprocessing.text import Tokenizer

# Assuming the maximum number of features your model was trained on
max_features = 3000 

# Function to preprocess input data for spam detection
def preprocess_data_for_spam(email_text):
    # Initialize the tokenizer
    tokenizer = Tokenizer(num_words=max_features)

    # Fit the tokenizer on the email text
    tokenizer.fit_on_texts([email_text])

    # Convert text to a sequence of integers
    email_seq = tokenizer.texts_to_sequences([email_text])

    # Flatten the list of lists to a single list
    email_flat = [item for sublist in email_seq for item in sublist]

    # Pad the sequences to ensure consistent input size
    # Adjust padding based on how your model was trained (pre or post)
    email_padded = pad_sequences([email_flat], maxlen=max_features, padding='pre')[0]

    return email_padded

# Adjust the predict call in your Streamlit app
def detect_spam(data):
    processed_data = preprocess_data_for_spam(data)
    prediction = spam_model.predict(np.array([processed_data]))
    is_spam = (prediction > 0.5).astype(int)[0][0]
    return is_spam


# Streamlit application start
def main():
    # Set background color
    set_bg_color("#A7C7E7")
    
    with st.sidebar:
        st.title('Navigation')
        if st.button('Home'):
            st.session_state['page'] = 'home'
        if st.button('About'):
            st.session_state['page'] = 'about'
        if st.button('Models'):
            st.session_state['page'] = 'models'
        if st.button('Contact'):
            st.session_state['page'] = 'contact'
        with st.expander("Disclaimer"):
            st.write("No financial advice. The contents of this website are not legally binding. This website is a student project. ")


    # Initialize the page state if it doesn't exist
    if 'page' not in st.session_state:
        st.session_state['page'] = 'home'  # Set to 'home' by default

    # Now we check the state to determine what to display
    if st.session_state['page'] == 'home':
        st.title("Email Classifier")

        # Spam Detection Section
        st.subheader("Spam Detection:")
        user_email = st.text_area("Enter email text for spam detection:")
        if st.button("Detect Spam"):
            spam_result = detect_spam(user_email)
            st.write(f"Email is {'Spam' if spam_result else 'Not Spam'}")
            
        # Sentiment Analysis Section
        st.subheader("Sentiment Analysis:")
        user_input = st.text_area("Enter text here for sentiment analysis:")
        if st.button("Analyze Sentiment"):
            sentiment_result = analyze_sentiment(user_input)
            st.write(f"Sentiment: {sentiment_result}")
            
    if st.session_state['page'] == 'about':
        st.title("About the app")
        st.markdown("""
**Email Classifier** is designed to streamline the sorting and categorization of emails for financial businesses. Utilizing neural network algorithms, it efficiently processes and classifies emails by analyzing patterns found in extensive datasets of email communications and financial news. This smart tool enhances email management, ensuring relevant information is readily accessible.
""")
    if st.session_state['page'] == 'models':
        st.title("Models' Performances")
        st.subheader("Spam Detection:")
        st.markdown("###### Acurracy: 98.8% ")
        
        conf_matrix_values = [[564, 10], [3, 525]]
# Plot the spam detection confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix_values, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted labels', fontweight='bold')
        plt.ylabel('True labels', fontweight='bold')
        plt.xticks(ticks=np.arange(2) + 0.5, labels=['Spam', 'No spam'])
        plt.yticks(ticks=np.arange(2) + 0.5, labels=['Spam', 'No spam'])
        plt.title('Confusion Matrix', fontsize=16)
        st.pyplot(plt)
        
        st.subheader("Sentiment Analysis:")
        st.markdown("###### Acurracy: 75.4% ")
        conf_matrix_sentiment_values = [[263, 21, 20], [43, 162, 53], [21, 47, 204]]
# Plot the sentiment analysis confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix_sentiment_values, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Predicted labels', fontweight='bold')
        plt.ylabel('True labels', fontweight='bold')
        plt.xticks(ticks=np.arange(3) + 0.5, labels=['Negative', 'Neutral', 'Positive'])
        plt.yticks(ticks=np.arange(3) + 0.5, labels=['Negative', 'Neutral', 'Positive'])
        plt.title('Confusion Matrix', fontsize=16)
        st.pyplot(plt)
        
    
    if st.session_state['page'] == 'contact':
        st.title("Contact Us")
        st.markdown("We're here to help and answer any questions you might have!")
        st.markdown('**Email:** emailclassifier@gmail.com')
        st.markdown('**Phone:** +41 (0)22 157 16 21')
if __name__ == "__main__":
    main() 