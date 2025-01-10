import tkinter as tk
from tkinter import filedialog, messagebox


class Classifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Text Classifier")
        boldFont18 = ("Arial", 18, "bold")
        # Model Selection Section
        self.modelFrame = tk.Frame(self.root)
        self.modelFrame.pack(pady=20)
        self.chosenModel = tk.IntVar()
        self.chosenModel.set(1)
        # "TFIDF Logistic Regression {'C': 1.5, 'solver': 'lbfgs', 'max_iter': 1000, 'tol': 0.001}"
        self.firstModelLabel = tk.Label(self.modelFrame, text="Model 1", font=boldFont18)
        self.firstModelLabel.grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.firstModelRadioButton = tk.Radiobutton(self.modelFrame, variable=self.chosenModel, value=1)
        self.firstModelRadioButton.grid(row=0, column=1, padx=10, pady=5)
        # "BagOfWords Random Forest {'n_estimators': 1000, 'max_depth': None, 'min_samples_split': 5}"
        self.secondModelLabel = tk.Label(self.modelFrame, text="Model 2", font=boldFont18)
        self.secondModelLabel.grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.secondModelRadioButton = tk.Radiobutton(self.modelFrame, variable=self.chosenModel, value=2)
        self.secondModelRadioButton.grid(row=1, column=1, padx=10, pady=5)

        # File Input Section
        self.fileFrame = tk.Frame(self.root)
        self.fileFrame.pack(pady=20)
        self.fileLabel = tk.Label(self.fileFrame, text="Classify by File:", font=boldFont18)
        self.fileLabel.pack(side=tk.LEFT, padx=10)
        self.browseButton = tk.Button(self.fileFrame, text="Browse File", command=self.browseFile, font=boldFont18)
        self.browseButton.pack(side=tk.LEFT, padx=10)
        # Text Input Section
        self.textFrame = tk.Frame(self.root)
        self.textFrame.pack(pady=20)
        self.textLabel = tk.Label(self.textFrame, text="Classify by Text Input:", font=boldFont18)
        self.textLabel.pack(anchor="w")
        self.textInput = tk.Text(self.textFrame, height=20, width=100, font=("Ariel", 16))
        self.textInput.pack(pady=10)

        self.classifyButton = tk.Button(self.root, text="Classify Text", command=self.classifyInputText, font=boldFont18)
        self.classifyButton.pack(pady=20)

    def browseFile(self):
        filePath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if filePath:
            print(f"File selected: {filePath}")
            # Placeholder for actual logic
            messagebox.showinfo("File Selected", f"Selected file: {filePath}")

    def classifyInputText(self):
        inputText = self.textInput.get("1.0", tk.END).strip()
        if inputText:
            print(f"Input text: {inputText}")
            # Placeholder for actual logic
            messagebox.showinfo("Classification Result", "Text classification logic placeholder.")
        else:
            print("No text entered.")
            messagebox.showwarning("Input Required", "Please enter some text to classify.")


if __name__ == "__main__":
    rootInterface = tk.Tk()
    app = Classifier(rootInterface)
    rootInterface.mainloop()
