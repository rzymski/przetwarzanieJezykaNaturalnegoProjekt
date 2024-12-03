import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def loadFilesFromFolder(inputFolder, maxNumberOfFiles=None):
    documents = []
    for subfolder in ['neg', 'pos']:
        folderPath = os.path.join(inputFolder, subfolder)
        print(f"📂 Rozpoczynanie wczytywania dokumentów z folderu: {folderPath}")
        for index, file in enumerate(os.listdir(folderPath), 1):
            if file.endswith('.txt'):
                filePath = os.path.join(folderPath, file)
                with open(filePath, 'r', encoding='utf-8') as f:
                    content = f.read()
                    documents.append(content)
                if maxNumberOfFiles and index >= maxNumberOfFiles:
                    print(f"✔️ Wczytano maksymalną liczbę plików ({maxNumberOfFiles}) z folderu: {folderPath}")
                    break
    print(f"✔️ Wczytano wszystkie dokumenty (łącznie: {len(documents)}).")
    return documents


def saveInChunks(matrix, outputPath, chunkSize=5000):
    if os.path.exists(outputPath):
        os.remove(outputPath)
        print(f"🗑️ Usunięto istniejący plik: {outputPath}")
    print(f"💾 Zapisuję macierz do pliku {outputPath} w partiach po {chunkSize} wierszy...")
    num_rows = matrix.shape[0]
    for start in range(0, num_rows, chunkSize):
        end = min(start + chunkSize, num_rows)
        chunk = matrix[start:end].toarray()
        pd.DataFrame(chunk).to_csv(outputPath, mode='a', header=False, index=False)
        print(f"✅ Zapisano wiersze od {start} do {end}.")


def createVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput, chunkSize=1000, maxNumberOfFiles=None):
    # Wczytaj dokumenty z folderów treningowych i testowych
    print("🔄 Rozpoczynanie wczytywania dokumentów treningowych i testowych...")
    trainDocuments = loadFilesFromFolder(trainFolder, maxNumberOfFiles)
    testDocuments = loadFilesFromFolder(testFolder, maxNumberOfFiles)
    # Oblicz macierz TF-IDF
    print("🔧 Rozpoczynanie obliczania macierzy TF-IDF...")
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")    
    trainTfidfMatrix = vectorizer.fit_transform(trainDocuments)
    testTfidfMatrix = vectorizer.transform(testDocuments)
    print("✅ Obliczono macierz TF-IDF.")
    # Zapisuj macierze TF-IDF partiami do plików CSV
    saveInChunks(trainTfidfMatrix, trainOutput, chunkSize)
    saveInChunks(testTfidfMatrix, testOutput, chunkSize)


if __name__ == "__main__":
    trainFolderPath = 'dataProcessed/train'
    testFolderPath = 'dataProcessed/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'
    print("🚀 Rozpoczynanie procesu tworzenia reprezentacji wektorowych...")
    createVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile, maxNumberOfFiles=1000)
    print("✔️ Proces tworzenia reprezentacji wektorowych zakończony.")
