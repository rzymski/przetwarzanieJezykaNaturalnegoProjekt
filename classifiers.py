import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


def loadVectorizedData(trainFile, testFile):
    """Wczytuje macierze TF-IDF z plików CSV wraz z etykietami."""
    print(f"📂 Wczytywanie danych z plików:\n- Trening: {trainFile}\n- Test: {testFile}")
    # Wczytaj dane treningowe
    train_data = pd.read_csv(trainFile, header=None)
    X_train = train_data.iloc[:, :-1].values  # Wszystkie kolumny oprócz ostatniej to cechy
    y_train = train_data.iloc[:, -1].values   # Ostatnia kolumna to etykiety

    # Wczytaj dane testowe
    test_data = pd.read_csv(testFile, header=None)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    print(f"✅ Wczytano dane: {X_train.shape[0]} treningowych, {X_test.shape[0]} testowych.")
    return X_train, y_train, X_test, y_test


def trainAndEvaluateClassifiers(X_train, y_train, X_test, y_test):
    """Trenuje i ocenia różne klasyfikatory."""
    classifiers = {
        "SVM": LinearSVC(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": MultinomialNB()
    }

    results = {}
    for name, clf in classifiers.items():
        print(f"🚀 Trenowanie klasyfikatora: {name}")
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Ocena modelu
        accuracy = accuracy_score(y_test, y_pred)
        print(f"📊 Wyniki klasyfikatora {name}:")
        print(classification_report(y_test, y_pred))

        # Zapis wyników
        results[name] = {
            "classifier": clf,
            "accuracy": accuracy,
            "report": classification_report(y_test, y_pred, output_dict=True)
        }

    return results


if __name__ == "__main__":
    # Ścieżki do danych
    trainMatrixFile = 'trainingMatrix.csv'
    testMatrixFile = 'testMatrix.csv'

    # Wczytaj dane z plików
    print("🔄 Wczytywanie danych z macierzy TF-IDF...")
    X_train, y_train, X_test, y_test = loadVectorizedData(trainMatrixFile, testMatrixFile)

    # Trenowanie i ocena klasyfikatorów
    print("🚀 Rozpoczynanie treningu i oceny klasyfikatorów...")
    results = trainAndEvaluateClassifiers(X_train, y_train, X_test, y_test)

    # Wyświetlenie wyników
    print("🎉 Zakończono trening klasyfikatorów. Wyniki:")
    for name, result in results.items():
        print(f"{name}: Dokładność: {result['accuracy']:.2f}")

    # Zapis najlepszego modelu
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])[1]['classifier']
    pd.to_pickle(best_model, "best_model.pkl")
    print("✅ Najlepszy model zapisany jako 'best_model.pkl'.")
