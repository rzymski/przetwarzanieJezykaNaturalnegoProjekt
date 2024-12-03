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


def saveInChunks(matrix, outputPath, chunk_size=3000):
    print(f"💾 Zapisuję macierz do pliku {outputPath} w partiach po {chunk_size} wierszy...")
    num_rows = matrix.shape[0]
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = matrix[start:end].toarray()
        pd.DataFrame(chunk).to_csv(outputPath, mode='a', header=False, index=False)
        print(f"✅ Zapisano wiersze od {start} do {end}.")


def createVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput, chunk_size=5000):
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

    # Zapisuj macierze TF-IDF partiami do plików CSV
    saveInChunks(trainTfidfMatrix, trainOutput, chunk_size)
    saveInChunks(testTfidfMatrix, testOutput, chunk_size)


if __name__ == "__main__":
    trainFolderPath = 'dataProcessed/train'
    testFolderPath = 'dataProcessed/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'

    print("🚀 Rozpoczynanie procesu tworzenia reprezentacji wektorowych...")
    createVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile)
    print("✔️ Proces tworzenia reprezentacji wektorowych zakończony.")
