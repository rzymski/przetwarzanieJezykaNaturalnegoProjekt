import tkinter as tk
from tkinter import filedialog, messagebox
import configparser
from utils import loadFromPkl
from classifier import classifySentiment


class Classifier:
    def __init__(self, root, configFile):
        self.root = root
        self.root.title("Text Classifier")
        boldFont18 = ("Arial", 18, "bold")
        # Load all models, vectors, and PCA files at startup
        self.models = self.loadAllModels(configFile)
        firstModelKey = next(iter(self.models))
        self.currentModel = self.models[firstModelKey]
        # Model Selection Section (Dropdown)
        self.modelFrame = tk.Frame(self.root)
        self.modelFrame.pack(pady=10)
        self.modelLabel = tk.Label(self.modelFrame, text="Select Model:", font=boldFont18)
        self.modelLabel.pack(side=tk.LEFT, padx=10)
        # Using tk.OptionMenu
        self.selectedModelDropdown = tk.StringVar()
        self.selectedModelDropdown.set(firstModelKey)
        self.modelDropdown = tk.OptionMenu(self.modelFrame, self.selectedModelDropdown, *self.models.keys(), command=self.onModelSelect)
        self.modelDropdown.config(font=boldFont18, width=40)
        self.modelDropdown.pack(side=tk.LEFT, padx=10)
        # File Input Section
        self.fileFrame = tk.Frame(self.root)
        self.fileFrame.pack(pady=10)
        self.fileLabel = tk.Label(self.fileFrame, text="Classify by File:", font=boldFont18)
        self.fileLabel.pack(side=tk.LEFT, padx=10)
        self.browseButton = tk.Button(self.fileFrame, text="Browse File", command=self.browseFile, font=boldFont18)
        self.browseButton.pack(side=tk.LEFT, padx=10)
        # Text Input Section
        self.textFrame = tk.Frame(self.root)
        self.textFrame.pack(pady=10)
        self.textLabel = tk.Label(self.textFrame, text="Classify by Text Input:", font=boldFont18)
        self.textLabel.pack(anchor="w")
        self.textInput = tk.Text(self.textFrame, height=20, width=100, font=("Ariel", 16))
        self.textInput.pack(pady=(20, 0))
        # Classify button and result display
        self.classifyButton = tk.Button(self.root, text="Classify Text", command=self.classifyInputText, font=boldFont18)
        self.classifyButton.pack(side=tk.TOP, pady=(0, 10), anchor="center")  # Centered button
        # Result display with padding from the right
        self.resultLabel = tk.Label(self.root, text="", font=("Arial", 32, "bold"), anchor="e")
        self.resultLabel.place(relx=1.0, x=-50, y=720, anchor="e")  # Right-aligned with 50 padding from right

        self.root.update_idletasks()
        windowWidth = self.root.winfo_width()
        windowHeight = self.root.winfo_height()
        screenWidth = self.root.winfo_screenwidth()
        screenHeight = self.root.winfo_screenheight()
        print(f"{windowWidth=} {windowHeight=} {screenWidth=} {screenHeight=}")
        x = (screenWidth // 2) - (windowWidth // 2)
        y = (screenHeight // 2) - (windowHeight // 2)
        self.root.geometry(f"{windowWidth}x{windowHeight}+{x}+{y}")

    @staticmethod
    def loadAllModels(configFile):
        config = configparser.ConfigParser()
        config.read(configFile)
        models = {}
        for section in config.sections():
            try:
                modelData = {
                    "model": loadFromPkl(config[section].get("model", ""), debug=False),
                    "vector": loadFromPkl(config[section].get("vector", ""), debug=False),
                    "pca": loadFromPkl(config[section].get("pca", ""), debug=False) if config[section].get("pca") else None,
                }
                models[section] = modelData
                print(f"✅ Loaded all resources for {section}.")
            except Exception as e:
                print(f"❌ Error loading resources for {section}: {e}")
        return models

    def onModelSelect(self, modelName):
        if modelName in self.models:
            self.currentModel = self.models[modelName]
            print(f"Selected Model: {modelName}")
            print(f" - Model Object: {self.currentModel['model']}")
            print(f" - Vector Object: {self.currentModel['vector']}")
            print(f" - PCA Object: {self.currentModel['pca']}")

    def browseFile(self):
        filePath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
        if filePath:
            if filePath.endswith(".txt"):
                try:
                    with open(filePath, 'r', encoding='utf-8') as file:
                        content = file.read()
                    self.textInput.delete("1.0", tk.END)  # Clear the text input
                    self.textInput.insert(tk.END, content)  # Insert file content
                    # messagebox.showinfo("File Loaded", f"Successfully loaded the content from: {filePath}")
                except Exception as e:
                    messagebox.showerror("Error", f"An error occurred while reading the file:\n{e}")
            else:
                messagebox.showerror("Invalid File", "The selected file is not a .txt file. Please choose a valid text file.")

    def classifyInputText(self):
        inputText = self.textInput.get("1.0", tk.END).strip()
        if inputText:
            print(f"Input text: {inputText}")
            result = classifySentiment(inputText, self.currentModel['model'], self.currentModel['vector'], self.currentModel['pca'])
            self.resultLabel.config(text=f"{('POSITIVE' if result else 'NEGATIVE')} REVIEW", fg=('green' if result else 'red'))
            # messagebox.showinfo("Classification Result", f"Text was classified as {'positive' if result else 'negative'}.")
        else:
            print("No text entered.")
            self.resultLabel.config(text=f"")
            messagebox.showwarning("Input Required", "Please enter some text to classify.")


if __name__ == "__main__":
    rootInterface = tk.Tk()
    app = Classifier(rootInterface, "classifier-config.ini")
    rootInterface.mainloop()
