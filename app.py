import streamlit as st
import pickle as pkl
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from string import punctuation

vectorizer = pkl.load(open('vectorizer.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

def transform_text(text):

    # LOWER CASING WORDS
    text = text.lower()

    # TOKANIZING WORDS
    text = nltk.word_tokenize(text)

    # REMOVING SPECIAL CHARACTERS
    res = []
    for i in text:
        if i.isalnum():
            res.append(i)
    
    # REMOVING STOP WORDS AND PUNCTUATION
    text = res[:]
    res.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in punctuation:
            res.append(i)

    # STEMMING
    text = res[:]
    res.clear()

    ps = PorterStemmer()
    for i in text:
        res.append(ps.stem(i))
    
    return " ".join(res)


st.title('SMS SPAM CLASSIFIER')
text = st.text_input('Enter The Message')
if st.button('predict'):
    transformed_text = transform_text(text)
    vector = vectorizer.transform([transformed_text])
    predicted_value = model.predict(vector)[0]

    if predicted_value == 1:
        st.header('SPAM')
    else:
        st.header('NOT SPAM')