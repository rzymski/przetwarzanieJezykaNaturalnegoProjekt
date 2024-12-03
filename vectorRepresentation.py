import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def loadFilesFromFolder(inputFolder, maxNumberOfFiles=1000):
    documents = []
    print("📂 Rozpoczynanie wczytywania dokumentów z folderu:", inputFolder)
    for root, dirs, files in os.walk(inputFolder):
        for index, file in enumerate(files, 1):
            if file.endswith('.txt'):
                filePath = os.path.join(root, file)
                with open(filePath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                print(f"📄 Wczytano plik: {filePath}")
            if index >= maxNumberOfFiles:
                print(f"✔️ Wczytano {maxNumberOfFiles=} dokumenty.")
                return documents
    print("✔️ Wczytano wszystkie dokumenty.")
    return documents


def createVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput):
    # Wczytaj dokumenty z folderów treningowych i testowych
    print("🔄 Rozpoczynanie wczytywania dokumentów treningowych i testowych...")
    trainDocuments = loadFilesFromFolder(trainFolder)
    testDocuments = loadFilesFromFolder(testFolder)

    # Oblicz macierz TF-IDF
    print("🔧 Rozpoczynanie obliczania macierzy TF-IDF...")
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")
    trainTfidfMatrix = vectorizer.fit_transform(trainDocuments)
    testTfidfMatrix = vectorizer.transform(testDocuments)
    print("✅ Obliczono macierz TF-IDF.")

    # Konwertuj macierze TF-IDF do DataFrame i zapisz do plików CSV
    print("💾 Zapisuję macierze TF-IDF do plików CSV...")
    trainTfidfDF = pd.DataFrame(trainTfidfMatrix.toarray(), columns=vectorizer.get_feature_names_out())
    testTfidfDF = pd.DataFrame(testTfidfMatrix.toarray(), columns=vectorizer.get_feature_names_out())

    trainTfidfDF.to_csv(trainOutput, index=False, header=False)
    testTfidfDF.to_csv(testOutput, index=False, header=False)
    print(f"✅ Macierze TF-IDF zostały zapisane w plikach:\n{trainOutput}\n{testOutput}")


if __name__ == "__main__":
    trainFolderPath = 'dataProcessed/train'
    testFolderPath = 'dataProcessed/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'

    print("🚀 Rozpoczynanie procesu tworzenia reprezentacji wektorowych...")
    createVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile)
    print("✔️ Proces tworzenia reprezentacji wektorowych zakończony.")
