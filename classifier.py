import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import torch
from transformers import BertTokenizer, BertModel
from lemmatization import processText


# Initialize BERT tokenizer and model globally for efficiency
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


def generateBERTEmbedding(text):
    tokens = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**tokens)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # CLS token embedding
    return cls_embedding.numpy().flatten()


def textToWord2VecVector(processedText, word2VecModel):
    words = processedText.split()
    wordVectors = [word2VecModel.wv[word] for word in words if word in word2VecModel.wv]
    if not wordVectors:
        return np.zeros(word2VecModel.vector_size)
    return np.mean(wordVectors, axis=0)


def classifySentiment(text, model, vectorizer, reducer=None):
    # Lematacja tekstu
    processedText = processText(text) if isinstance(vectorizer, (TfidfVectorizer, CountVectorizer, Word2Vec)) else text
    # Przekształcenie tekstu na wektor
    if isinstance(vectorizer, Word2Vec):  # Jeśli to Word2Vec
        textVector = np.array([textToWord2VecVector(processedText, vectorizer)])
    elif isinstance(vectorizer, (TfidfVectorizer, CountVectorizer)):  # Jeśli to TF-IDF lub Bag of Words
        textVector = vectorizer.transform([processedText])
    elif vectorizer is None:  # Jeśli to BERT
        textVector = generateBERTEmbedding(text).reshape(1, -1)
    else:
        raise TypeError(f"Nieobsługiwany typ wektoryzera: {type(vectorizer)}")
    # Jeśli to Voting Classifier
    if hasattr(textVector, "toarray"):
        textVector = textVector.toarray()
    # Zastosowanie PCA (jeśli istnieje)
    if reducer:
        textVector = reducer.transform(textVector)
    # Predykcja
    prediction = model.predict(textVector)
    probabilities = model.predict_proba(textVector)[0] if hasattr(model, "predict_proba") else None
    return prediction[0] == 1, probabilities
