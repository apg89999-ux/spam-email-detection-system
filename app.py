import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title('📧 Spam Email Detection System')
st.write('Enter an email or message below to check whether it is spam.')

user_input = st.text_area('Enter Email Text')

if st.button('Check Spam'):
    input_tfidf = vectorizer.transform([user_input])
    prediction = model.predict(input_tfidf)[0]

    if prediction == 1:
        st.error('🚨 This message is SPAM')
    else:
        st.success('✅ This message is NOT SPAM')

st.write(f'### Model Accuracy: {accuracy*100:.2f}%')