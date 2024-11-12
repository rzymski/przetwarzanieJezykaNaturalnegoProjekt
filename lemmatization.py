import os
import spacy

nlp = spacy.load('en_core_web_sm')


def processText(text):  # Funkcja do lematyzacji tekstu
    doc = nlp(text)
    processedWords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return ' '.join(processedWords)


def processFilesInFolder(inputFolderPath, outputFolderPath):  # Funkcja do przetwarzania plików w folderze
    for root, dirs, files in os.walk(inputFolderPath):
        for file in files:
            if file.endswith(".txt"):
                # Pełna ścieżka do pliku wejściowego
                filePath = os.path.join(root, file)
                # Wczytanie tekstu z pliku
                with open(filePath, 'r', encoding='utf-8') as f:
                    text = f.read()
                # Przetworzenie tekstu (lematyzacja)
                processedText = processText(text)
                # Utworzenie ścieżki do zapisu przetworzonego pliku
                relativePath = os.path.relpath(filePath, inputFolderPath)
                outputFilePath = os.path.join(outputFolderPath, relativePath)
                # Utworzenie katalogu, jeśli nie istnieje
                os.makedirs(os.path.dirname(outputFilePath), exist_ok=True)
                # Zapisanie przetworzonego tekstu
                with open(outputFilePath, 'w', encoding='utf-8') as f:
                    f.write(processedText)


def lemmatizeData(inputDirectory, outputDirectory):
    # Przetworzenie plików w folderach train/pos, train/neg, test/pos, test/neg
    processFilesInFolder(os.path.join(inputDirectory, 'train/pos'), os.path.join(outputDirectory, 'train/pos'))
    processFilesInFolder(os.path.join(inputDirectory, 'train/neg'), os.path.join(outputDirectory, 'train/neg'))
    processFilesInFolder(os.path.join(inputDirectory, 'test/pos'), os.path.join(outputDirectory, 'test/pos'))
    processFilesInFolder(os.path.join(inputDirectory, 'test/neg'), os.path.join(outputDirectory, 'test/neg'))
    print("Przetwarzanie plików zakończone!")


if __name__ == "__main__":
    inputDir = 'data'
    outputDir = 'dataProcessed'
    lemmatizeData(inputDir, outputDir)
