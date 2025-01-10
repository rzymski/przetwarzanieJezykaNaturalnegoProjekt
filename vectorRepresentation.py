import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec, FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from utils import measureExecutionTime


def loadFilesFromFolder(inputFolder, maxNumberOfFiles=None):
    documents = []
    for subfolder in ['neg', 'pos']:
        folderPath = os.path.join(inputFolder, subfolder)
        # print(f"📂 Rozpoczynanie wczytywania dokumentów z folderu: {folderPath}")
        for index, file in enumerate(os.listdir(folderPath), 1):
            if file.endswith('.txt'):
                filePath = os.path.join(folderPath, file)
                with open(filePath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                if maxNumberOfFiles and index >= maxNumberOfFiles:
                    # print(f"✔️ Wczytano maksymalną liczbę plików ({maxNumberOfFiles}) z folderu: {folderPath}")
                    break
    # print(f"✔️ Wczytano wszystkie dokumenty (łącznie: {len(documents)}).")
    return documents


def saveInChunks(matrix, outputPath, chunkSize=5000):
    if os.path.exists(outputPath):
        os.remove(outputPath)
        # print(f"🗑️ Usunięto istniejący plik: {outputPath}")
    # print(f"💾 Zapisuję macierz do pliku {outputPath} w partiach po {chunkSize} wierszy...")
    num_rows = matrix.shape[0]
    for start in range(0, num_rows, chunkSize):
        end = min(start + chunkSize, num_rows)
        chunk = matrix[start:end].toarray()
        pd.DataFrame(chunk).to_csv(outputPath, mode='a', header=False, index=False)
        # print(f"✅ Zapisano wiersze od {start} do {end}.")


@measureExecutionTime
def createTFIDFVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput, chunkSize=1000, maxNumberOfFiles=None):
    print("🔄 Rozpoczynanie wczytywania dokumentów treningowych i testowych...")
    trainDocuments = loadFilesFromFolder(trainFolder, maxNumberOfFiles)
    testDocuments = loadFilesFromFolder(testFolder, maxNumberOfFiles)
    print("🔧 Rozpoczynanie obliczania macierzy TF-IDF...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=5, max_df=0.9, sublinear_tf=True, binary=True)
    trainTfidfMatrix = vectorizer.fit_transform(trainDocuments)
    testTfidfMatrix = vectorizer.transform(testDocuments)
    print("✅ Obliczono macierz TF-IDF.")
    saveInChunks(trainTfidfMatrix, trainOutput, chunkSize)
    saveInChunks(testTfidfMatrix, testOutput, chunkSize)
    return vectorizer


@measureExecutionTime
def createBagOfWordsVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput, chunkSize=1000, maxNumberOfFiles=None):
    print("🔄 Rozpoczynanie wczytywania dokumentów treningowych i testowych...")
    trainDocuments = loadFilesFromFolder(trainFolder, maxNumberOfFiles)
    testDocuments = loadFilesFromFolder(testFolder, maxNumberOfFiles)
    print("🔧 Rozpoczynanie obliczania macierzy Count Vectorizer...")
    vectorizer = CountVectorizer(ngram_range=(1, 3), max_features=10000, min_df=5, max_df=0.9, binary=True)
    trainCountMatrix = vectorizer.fit_transform(trainDocuments)
    testCountMatrix = vectorizer.transform(testDocuments)
    print("✔️ Obliczono macierz Count Vectorizer.")
    saveInChunks(trainCountMatrix, trainOutput, chunkSize)
    saveInChunks(testCountMatrix, testOutput, chunkSize)
    return vectorizer


@measureExecutionTime
def createWord2VecVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput, vectorSize=300, window=10, minCount=3, workers=4, epochs=50, maxNumberOfFiles=None):
    print("🔄 Rozpoczynanie wczytywania dokumentów treningowych i testowych do Word2Vec...")
    trainDocuments = loadFilesFromFolder(trainFolder, maxNumberOfFiles)
    testDocuments = loadFilesFromFolder(testFolder, maxNumberOfFiles)
    print("🔧 Tokenizacja dokumentów...")
    tokenizedTrain = [simple_preprocess(doc) for doc in trainDocuments]
    tokenizedTest = [simple_preprocess(doc) for doc in testDocuments]
    print("📚 Trening modelu Word2Vec...")
    word2vecModel = Word2Vec(sentences=tokenizedTrain, vector_size=vectorSize, window=window, min_count=minCount, workers=workers)
    print("🔄 Dodatkowe iteracje treningu Word2Vec...")
    word2vecModel.train(tokenizedTrain, total_examples=len(tokenizedTrain), epochs=epochs)
    def documentToVector(tokens, model):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vectorSize)
    print("🔄 Konwersja dokumentów na macierze wektorów Word2Vec...")
    trainVectors = [documentToVector(doc, word2vecModel) for doc in tokenizedTrain]
    testVectors = [documentToVector(doc, word2vecModel) for doc in tokenizedTest]
    print("💾 Zapis wektorów Word2Vec do plików CSV...")
    pd.DataFrame(trainVectors).to_csv(trainOutput, index=False, header=False)
    pd.DataFrame(testVectors).to_csv(testOutput, index=False, header=False)
    print("✔️ Zakończono zapis wektorów Word2Vec.")
    return word2vecModel


@measureExecutionTime
def createDoc2VecVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput, vectorSize=300, window=10, minCount=3, workers=4, epochs=50, maxNumberOfFiles=None):
    print("🔄 Rozpoczynanie wczytywania dokumentów treningowych i testowych do Doc2Vec...")
    trainDocuments = loadFilesFromFolder(trainFolder, maxNumberOfFiles)
    testDocuments = loadFilesFromFolder(testFolder, maxNumberOfFiles)
    print("🔧 Tokenizacja i oznaczanie dokumentów...")
    taggedTrain = [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in enumerate(trainDocuments)]
    print("📚 Trening modelu Doc2Vec...")
    doc2vecModel = Doc2Vec(vector_size=vectorSize, window=window, min_count=minCount, workers=workers, epochs=epochs)
    doc2vecModel.build_vocab(taggedTrain)
    doc2vecModel.train(taggedTrain, total_examples=len(taggedTrain), epochs=epochs)
    print("🔄 Generowanie wektorów dla dokumentów...")
    trainVectors = [doc2vecModel.infer_vector(simple_preprocess(doc)) for doc in trainDocuments]
    testVectors = [doc2vecModel.infer_vector(simple_preprocess(doc)) for doc in testDocuments]
    print("💾 Zapis wektorów Doc2Vec do plików CSV...")
    pd.DataFrame(trainVectors).to_csv(trainOutput, index=False, header=False)
    pd.DataFrame(testVectors).to_csv(testOutput, index=False, header=False)
    print("✔️ Zakończono zapis wektorów Doc2Vec.")
    return doc2vecModel


@measureExecutionTime
def createFastTextVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput, vectorSize=300, window=5, minCount=3, workers=4, epochs=50, maxNumberOfFiles=None):
    print("🔄 Rozpoczynanie wczytywania dokumentów treningowych i testowych do FastText...")
    trainDocuments = loadFilesFromFolder(trainFolder, maxNumberOfFiles)
    testDocuments = loadFilesFromFolder(testFolder, maxNumberOfFiles)
    print("🔧 Tokenizacja dokumentów...")
    tokenizedTrain = [simple_preprocess(doc) for doc in trainDocuments]
    tokenizedTest = [simple_preprocess(doc) for doc in testDocuments]
    print("📚 Trening modelu FastText...")
    fastTextModel = FastText(sentences=tokenizedTrain, vector_size=vectorSize, window=window, min_count=minCount, workers=workers)
    fastTextModel.train(tokenizedTrain, total_examples=len(tokenizedTrain), epochs=epochs)
    def documentToVector(tokens, model):
        vectors = [model.wv[token] for token in tokens if token in model.wv]
        if vectors:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(vectorSize)
    print("🔄 Konwersja dokumentów na macierze wektorów FastText...")
    trainVectors = [documentToVector(doc, fastTextModel) for doc in tokenizedTrain]
    testVectors = [documentToVector(doc, fastTextModel) for doc in tokenizedTest]
    print("📄 Zapis wektorów FastText do plików CSV...")
    pd.DataFrame(trainVectors).to_csv(trainOutput, index=False, header=False)
    pd.DataFrame(testVectors).to_csv(testOutput, index=False, header=False)
    print("✔️ Zakończono zapis wektorów FastText.")
    return fastTextModel
