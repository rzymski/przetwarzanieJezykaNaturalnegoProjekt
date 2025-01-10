import os
import spacy

nlp = spacy.load('en_core_web_sm')


def processText(text):  # Funkcja do lematyzacji tekstu
    text = text.lower()
    doc = nlp(text)
    processedWords = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    # print("✅ Lematyzacja zakończona.")
    return ' '.join(processedWords)


def processFilesInFolder(inputFolderPath, outputFolderPath):  # Funkcja do przetwarzania plików w folderze
    print(f"🔄 Rozpoczynanie przetwarzania plików w folderze: {inputFolderPath}")
    for root, dirs, files in os.walk(inputFolderPath):
        for file in files:
            if file.endswith(".txt"):
                filePath = os.path.join(root, file)
                # print(f"📄 Przetwarzanie pliku: {filePath}")
                try:        
                    with open(filePath, 'r', encoding='utf-8') as f:
                        text = f.read()
                    processedText = processText(text)
                    relativePath = os.path.relpath(filePath, inputFolderPath)
                    outputFilePath = os.path.join(outputFolderPath, relativePath)
                    os.makedirs(os.path.dirname(outputFilePath), exist_ok=True)
                    with open(outputFilePath, 'w', encoding='utf-8') as f:
                        f.write(processedText)
                    # print(f"✅ Plik zapisany: {outputFilePath}")
                except Exception as e:
                    print(f"❌ Błąd podczas przetwarzania pliku {filePath}: {e}")
    print(f"✔️ Zakończono przetwarzanie plików w folderze: {inputFolderPath}")


def lemmatizeData(inputDirectory, outputDirectory):
    # Przetwarzanie plików w folderach train/pos, train/neg, test/pos, test/neg
    print(f"🚀 Rozpoczynanie lematyzacji danych w katalogu: {inputDirectory}")
    processFilesInFolder(os.path.join(inputDirectory, 'train/pos'), os.path.join(outputDirectory, 'train/pos'))
    processFilesInFolder(os.path.join(inputDirectory, 'train/neg'), os.path.join(outputDirectory, 'train/neg'))
    processFilesInFolder(os.path.join(inputDirectory, 'test/pos'), os.path.join(outputDirectory, 'test/pos'))
    processFilesInFolder(os.path.join(inputDirectory, 'test/neg'), os.path.join(outputDirectory, 'test/neg'))
    print("🎉 Przetwarzanie plików zakończone!")


if __name__ == "__main__":
    inputDir = 'data'
    outputDir = 'dataProcessed'
    print("🔧 Rozpoczynanie procesu lematyzacji danych.")
    lemmatizeData(inputDir, outputDir)
    print("✔️ Zakończono proces lematyzacji danych.")
