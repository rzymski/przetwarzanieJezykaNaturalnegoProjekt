import time
import os
import pickle


def measureExecutionTime(func):
    def wrapper(*args, **kwargs):
        startTime = time.time()
        result = func(*args, **kwargs)
        endTime = time.time()
        executionTime = endTime - startTime
        print(f"⏱️ Funkcja '{func.__name__}' wykonana w {executionTime:.4f} sekund.")
        return result
    return wrapper


def saveToPkl(obj, filePath, debug=True):
    try:
        os.makedirs(os.path.dirname(filePath), exist_ok=True)
        with open(filePath, 'wb') as file:
            pickle.dump(obj, file)
        if debug:
            print(f"✅ Obiekt zapisany w pliku: {filePath}")
    except Exception as e:
        print(f"❌ Błąd podczas zapisywania do pliku {filePath}: {e}")


def loadFromPkl(filePath, debug=True):
    try:
        with open(filePath, 'rb') as file:
            obj = pickle.load(file)
        if debug:
            print(f"✅ Obiekt wczytany z pliku: {filePath}")
        return obj
    except FileNotFoundError:
        print(f"❌ Błąd: Plik {filePath} nie istnieje.")
    except Exception as e:
        print(f"❌ Błąd podczas wczytywania pliku {filePath}: {e}")
    return None
