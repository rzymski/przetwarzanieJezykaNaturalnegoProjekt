import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def loadFilesFromFolder(inputFolder, maxNumberOfFiles=1000):
    documents = []
    labels = []  # Lista etykiet
    print("📂 Rozpoczynanie wczytywania dokumentów z folderu:", inputFolder)

    # Ścieżki do podfolderów `pos` i `neg`
    posFolder = os.path.join(inputFolder, 'pos')
    negFolder = os.path.join(inputFolder, 'neg')

    # Liczba plików do wczytania z każdej klasy
    maxFilesPerClass = maxNumberOfFiles // 2

    def readFilesFromClass(folder, label, maxFiles):
        class_documents = []
        class_labels = []
        count = 0
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.txt'):
                    filePath = os.path.join(root, file)
                    with open(filePath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        class_documents.append(content)
                        class_labels.append(label)
                    count += 1
                    print(f"📄 Wczytano plik: {filePath}")
                    if count >= maxFiles:
                        break
            if count >= maxFiles:
                break
        return class_documents, class_labels

    # Wczytaj pliki z klasy `pos`
    print("🔄 Wczytywanie dokumentów z klasy pozytywnej (pos)...")
    pos_documents, pos_labels = readFilesFromClass(posFolder, 1, maxFilesPerClass)

    # Wczytaj pliki z klasy `neg`
    print("🔄 Wczytywanie dokumentów z klasy negatywnej (neg)...")
    neg_documents, neg_labels = readFilesFromClass(negFolder, 0, maxFilesPerClass)

    # Połącz wyniki
    documents.extend(pos_documents)
    documents.extend(neg_documents)
    labels.extend(pos_labels)
    labels.extend(neg_labels)

    print(f"✔️ Wczytano dokumenty: {len(documents)} (pos: {len(pos_documents)}, neg: {len(neg_documents)})")
    return documents, labels


def saveInChunks(matrix, labels, outputPath, chunk_size=3000):
    print(f"💾 Zapisuję macierz do pliku {outputPath} w partiach po {chunk_size} wierszy...")
    num_rows = matrix.shape[0]
    for start in range(0, num_rows, chunk_size):
        end = min(start + chunk_size, num_rows)
        chunk = matrix[start:end].toarray()
        chunk_with_labels = pd.DataFrame(chunk)
        chunk_with_labels['label'] = labels[start:end]  # Dodaj etykiety jako ostatnią kolumnę
        chunk_with_labels.to_csv(outputPath, mode='a', header=False, index=False)
        print(f"✅ Zapisano wiersze od {start} do {end}.")


def createVectorRepresentations(trainFolder, testFolder, trainOutput, testOutput, chunk_size=5000):
    # Wczytaj dokumenty i etykiety z folderów treningowych i testowych
    print("🔄 Rozpoczynanie wczytywania dokumentów treningowych i testowych...")
    trainDocuments, trainLabels = loadFilesFromFolder(trainFolder)
    testDocuments, testLabels = loadFilesFromFolder(testFolder)

    # Oblicz macierz TF-IDF
    print("🔧 Rozpoczynanie obliczania macierzy TF-IDF...")
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w\w+\b")    
    trainTfidfMatrix = vectorizer.fit_transform(trainDocuments)
    testTfidfMatrix = vectorizer.transform(testDocuments)
    print("✅ Obliczono macierz TF-IDF.")

    # Zapisuj macierze TF-IDF wraz z etykietami do plików CSV
    saveInChunks(trainTfidfMatrix, trainLabels, trainOutput, chunk_size)
    saveInChunks(testTfidfMatrix, testLabels, testOutput, chunk_size)


if __name__ == "__main__":
    trainFolderPath = 'dataProcessed/train'
    testFolderPath = 'dataProcessed/test'
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'

    print("🚀 Rozpoczynanie procesu tworzenia reprezentacji wektorowych...")
    createVectorRepresentations(trainFolderPath, testFolderPath, trainMatrixFile, testMatrixFile)
    print("✔️ Proces tworzenia reprezentacji wektorowych zakończony.")
