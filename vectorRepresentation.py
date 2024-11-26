import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def loadFilesFromFolder(inputFolder):
    documents = []
    for root, dirs, files in os.walk(inputFolder):
        for file in files:
            if file.endswith('.txt'):
                filePath = os.path.join(root, file)
                with open(filePath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
    return documents


def createVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput):
    # Wczytaj dokumenty z folderów treningowych i testowych
    trainDocuments = loadFilesFromFolder(trainFolder)
    testDocuments = loadFilesFromFolder(testFolder)

    # Oblicz macierz TF-IDF
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
    trainTfidfMatrix = vectorizer.fit_transform(trainDocuments)
    testTfidfMatrix = vectorizer.transform(testDocuments)

    # Konwertuj macierze TF-IDF do DataFrame i zapisz do plików CSV
    trainTfidfDF = pd.DataFrame(trainTfidfMatrix.toarray(), columns=vectorizer.get_feature_names_out())
    testTfidfDF = pd.DataFrame(testTfidfMatrix.toarray(), columns=vectorizer.get_feature_names_out())

    trainTfidfDF.to_csv(trainOutput, index=False, header=False)
    testTfidfDF.to_csv(testOutput, index=False, header=False)

    print(f"Macierze TF-IDF zostały zapisane w plikach:\n{trainOutput}\n{testOutput}")


if __name__ == "__main__":
    trainFolderPath = 'dataProcessed/train'
    testFolderPath = 'dataProcessed/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'

    createVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile)
