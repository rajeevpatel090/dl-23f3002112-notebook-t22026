# 🧠 AI MCQ Solver

A Deep Learning based Multiple Choice Question (MCQ) Solver — given a question and its options, it **predicts the correct answer** among 5 options (A–E).

🔗 **Live Demo:** [rajeev5944-mcq-solver.hf.space](https://rajeev5944-mcq-solver.hf.space/)

This project was built as part of the **DLGenAI (T22026)** course milestone assignment, where a model is trained/fine-tuned and deployed through a **Gradio web app**.

---

## 📌 Overview

The goal of this project is simple — give it any MCQ question (with 5 options: A, B, C, D, E), and the model will predict which option is the correct answer.

Multiple approaches were explored (such as a scratch model, pretrained embeddings, and a fine-tuned transformer), and the best performing one is used as the final model. The entire pipeline is documented in the notebooks, and the final model is saved in the `models/` folder.

---

## ✨ Features

- ✅ Multiple Choice Question Answering (5 options: A–E)
- ✅ Deep Learning based model (Transformer / fine-tuned architecture)
- ✅ Simple and clean web UI built with Gradio
- ✅ Live deployed on Hugging Face Spaces
- ✅ End-to-end notebook workflow (data → training → evaluation)

---

## 📂 Project Structure

```
AI_MCQ-solver/
├── models/            # Trained / fine-tuned model files
├── notebooks/          # Data exploration, training & evaluation notebooks
├── milestone-2.ipynb   # Main milestone notebook (model building & results)
├── LICENSE              # MIT License
└── README.md            # Project documentation (this file)
```

---

## 🛠️ Tech Stack

| Category        | Tools / Libraries              |
|------------------|----------------------------------|
| Language          | Python                           |
| Deep Learning     | PyTorch, Hugging Face Transformers |
| Web Interface     | Gradio                           |
| Deployment        | Hugging Face Spaces              |
| Environment       | Jupyter Notebook                 |

---

## 🚀 Getting Started (Local Setup)

To clone the repo and run it on your local machine:

```bash
# 1. Clone the repository
git clone https://github.com/rajeevpatel090/AI_MCQ-solver.git
cd AI_MCQ-solver

# 2. Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch transformers gradio pandas numpy scikit-learn

# 4. Run the notebook
jupyter notebook milestone-2.ipynb
```

> **Note:** Adding a `requirements.txt` file to the repo will simplify installation — just run `pip install -r requirements.txt`.

---

## 📊 Model Approach

The project generally follows these steps:

1. **Data Preparation** – Cleaning and structuring the MCQ dataset.
2. **Model Building** – Trying different approaches (baseline ML model, pretrained embeddings + classifier, fine-tuned transformer).
3. **Evaluation** – Comparing models using metrics like MAP@3 / Accuracy.
4. **Deployment** – Deploying the best model with a Gradio interface on Hugging Face Spaces.

Detailed training and evaluation steps can be found in `milestone-2.ipynb` and the `notebooks/` folder.

---

## 🤝 Contributing

For suggestions or improvements:

1. **Fork** the repo
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m "Add your feature"`)
4. Push the branch (`git push origin feature/your-feature`)
5. Open a **Pull Request**

---

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for details.

---

## 👤 Author

**Rajeev Patel**
🔗 GitHub: [@rajeevpatel090](https://github.com/rajeevpatel090)
🔗 Live App: [rajeev5944-mcq-solver.hf.space](https://rajeev5944-mcq-solver.hf.space/)

---

⭐ If you found this project useful, don't forget to **star** the repo!
