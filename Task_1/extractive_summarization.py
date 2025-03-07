# Importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Downloading the NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Extractive Summarization Function
def extractive_summarization(text, summary_length=3):
    # Tokenizing sentences
    sentences = sent_tokenize(text)
    
    # Remove: stopwords and punctuation
    stop_words = set(stopwords.words('english'))
    processed_sentences = [
        ' '.join([word for word in word_tokenize(sentence.lower()) if word.isalnum() and word not in stop_words])
        for sentence in sentences
    ]
    
    # Now Computing TF-IDF scores
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_sentences)
    
    # Calculate sentence scores
    sentence_scores = np.sum(tfidf_matrix.toarray(), axis=1)
    
    # Now ranking sentences by score
    ranked_sentences = [sentence for _, sentence in sorted(zip(sentence_scores, sentences), reverse=True)]
    
    # Select top N sentences for the summary
    summary = ' '.join(ranked_sentences[:summary_length])
    return summary

# INPUT
text = """
The rapid growth of online shopping has significantly impacted traditional brick-and-mortar retail, with many consumers now preferring the convenience of purchasing goods from the comfort of their homes. This shift has led to a surge in e-commerce platforms, offering a vast array of products with competitive prices and fast delivery options, putting pressure on physical stores to adapt and integrate digital strategies to remain relevant in the modern marketplace. As a result, businesses are increasingly focusing on omnichannel experiences, allowing customers to seamlessly transition between online and offline shopping, including features like in-store pickup and product availability checks, to cater to evolving consumer behavior.
"""

summary = extractive_summarization(text, summary_length=2)
print("Summary:\n", summary)