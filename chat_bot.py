#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt')

data = pd.read_csv('q_and_a.csv')

data = data[['Question', 'Answer']]

data = data.dropna()

data['Question'] = data['Question'].str.lower()
data['Answer'] = data['Answer'].str.lower()

data['Question'] = data['Question'].apply(word_tokenize)
data['Answer'] = data['Answer'].apply(word_tokenize)

data['Question'] = data['Question'].apply(lambda x: ' '.join(x))
data['Answer'] = data['Answer'].apply(lambda x: ' '.join(x))

corpus = data['Question'].tolist()

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

def get_response(user_input):
    user_input = user_input.lower()
    user_input = ' '.join(word_tokenize(user_input))
    
    user_tfidf = vectorizer.transform([user_input])
    
    similarities = cosine_similarity(user_tfidf, tfidf_matrix)
    
    max_similarity_index = np.argmax(similarities)
    
    if similarities[0, max_similarity_index] > 0.5:  
        return data['Answer'].iloc[max_similarity_index]
    else:
        return "I'm sorry, I don't know the answer to that question."

# Chatbot function
def chatbot():
    print("Hello! I'm a retrieval-based chatbot. Ask me a question, or type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        answer = get_response(user_input)
        print(f"Chatbot: {answer}")

# Run the chatbot
if __name__ == "__main__":
    chatbot()


# In[ ]:




