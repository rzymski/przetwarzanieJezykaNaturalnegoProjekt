import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from utils import loadFromPkl
from lemmatization import processText


def textToWord2VecVector(processedText, word2VecModel):
    words = processedText.split()
    wordVectors = [word2VecModel.wv[word] for word in words if word in word2VecModel.wv]
    if not wordVectors:
        return np.zeros(word2VecModel.vector_size)
    return np.mean(wordVectors, axis=0)


def classifySentiment(inputTextOrPath, modelFilePath, vectorizerFilePath, pcaFilePath=None, isFile=True):
    # Wczytanie tekstu
    if isFile:
        with open(inputTextOrPath, 'r', encoding='utf-8') as file:
            text = file.read()
    else:
        text = inputTextOrPath
    # Lematacja tekstu
    processedText = processText(text)
    # Wczytanie modelu i wektoryzera
    classifier = loadFromPkl(modelFilePath)
    vectorizer = loadFromPkl(vectorizerFilePath)
    pca = loadFromPkl(pcaFilePath) if pcaFilePath else None
    # Przekształcenie tekstu na wektor
    if isinstance(vectorizer, Word2Vec):  # Jeśli to Word2Vec
        textVector = np.array([textToWord2VecVector(processedText, vectorizer)])
    elif isinstance(vectorizer, (TfidfVectorizer, CountVectorizer)):  # Jeśli to TF-IDF lub Bag of Words
        textVector = vectorizer.transform([processedText])
    else:
        raise TypeError(f"Nieobsługiwany typ wektoryzera: {type(vectorizer)}")
    # Zastosowanie PCA (jeśli istnieje)
    if pca:
        textVector = pca.transform(textVector)
    # Predykcja
    prediction = classifier.predict(textVector)
    return prediction[0] == 1


if __name__ == "__main__":
    tfidfResult = classifySentiment("reviewTestP.txt", "models/TFIDF_LogisticRegression.pkl", "vectors/TFIDF.pkl", "vectors/TFIDF_PCA.pkl")
    print(f"Recenzja jest: {'pozytywna' if tfidfResult else 'negatywna'}")
    tfidfResult = classifySentiment("reviewTestN.txt", "models/TFIDF_LogisticRegression.pkl", "vectors/TFIDF.pkl", "vectors/TFIDF_PCA.pkl")
    print(f"Recenzja jest: {'pozytywna' if tfidfResult else 'negatywna'}")
    tfidfResult = classifySentiment("reviewTestP.txt", "models/TFIDF_LogisticRegression2.pkl", "vectors/TFIDF.pkl", "vectors/TFIDF_PCA.pkl")
    print(f"Recenzja jest: {'pozytywna' if tfidfResult else 'negatywna'}")
    tfidfResult = classifySentiment("reviewTestN.txt", "models/TFIDF_LogisticRegression2.pkl", "vectors/TFIDF.pkl", "vectors/TFIDF_PCA.pkl")
    print(f"Recenzja jest: {'pozytywna' if tfidfResult else 'negatywna'}")

    bagOfWordsResult = classifySentiment("reviewTestP.txt", "models/BagOfWords_LogisticRegression.pkl", "vectors/BagOfWords.pkl", "vectors/BagOfWords_PCA.pkl")
    print(f"Recenzja jest: {'pozytywna' if bagOfWordsResult else 'negatywna'}")
    bagOfWordsResult = classifySentiment("reviewTestN.txt", "models/BagOfWords_LogisticRegression.pkl", "vectors/BagOfWords.pkl", "vectors/BagOfWords_PCA.pkl")
    print(f"Recenzja jest: {'pozytywna' if bagOfWordsResult else 'negatywna'}")
    bagOfWordsResult = classifySentiment("reviewTestP.txt", "models/BagOfWords_RandomForest.pkl", "vectors/BagOfWords.pkl")
    print(f"Recenzja jest: {'pozytywna' if bagOfWordsResult else 'negatywna'}")
    bagOfWordsResult = classifySentiment("reviewTestN.txt", "models/BagOfWords_RandomForest.pkl", "vectors/BagOfWords.pkl")
    print(f"Recenzja jest: {'pozytywna' if bagOfWordsResult else 'negatywna'}")

    world2VecResult = classifySentiment("reviewTestP.txt", "models/Word2Vec_LogisticRegression.pkl", "vectors/Word2Vec.pkl")
    print(f"Recenzja jest: {'pozytywna' if world2VecResult else 'negatywna'}")
    world2VecResult = classifySentiment("reviewTestN.txt", "models/Word2Vec_LogisticRegression.pkl", "vectors/Word2Vec.pkl")
    print(f"Recenzja jest: {'pozytywna' if world2VecResult else 'negatywna'}")
