import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from lemmatization import processText


def textToWord2VecVector(processedText, word2VecModel):
    words = processedText.split()
    wordVectors = [word2VecModel.wv[word] for word in words if word in word2VecModel.wv]
    if not wordVectors:
        return np.zeros(word2VecModel.vector_size)
    return np.mean(wordVectors, axis=0)


def classifySentiment(text, model, vectorizer, pca=None):
    # Lematacja tekstu
    processedText = processText(text)
    # Przekształcenie tekstu na wektor
    if isinstance(vectorizer, Word2Vec):  # Jeśli to Word2Vec
        textVector = np.array([textToWord2VecVector(processedText, vectorizer)])
    elif isinstance(vectorizer, (TfidfVectorizer, CountVectorizer)):  # Jeśli to TF-IDF lub Bag of Words
        textVector = vectorizer.transform([processedText])
    else:
        raise TypeError(f"Nieobsługiwany typ wektoryzera: {type(vectorizer)}")
    # Jeśli to Voting Classifier
    if hasattr(textVector, "toarray"):
        textVector = textVector.toarray()
    # Zastosowanie PCA (jeśli istnieje)
    if pca:
        textVector = pca.transform(textVector)
    # Predykcja
    prediction = model.predict(textVector)
    probabilities = model.predict_proba(textVector)[0] if hasattr(model, "predict_proba") else None
    return prediction[0] == 1, probabilities
