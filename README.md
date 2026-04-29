# 🧠 NLP Text Processing Pipeline

A natural language processing pipeline that takes a Spanish text input and runs it through **translation**, **summarization**, **sentiment analysis**, and a custom **LSTM-based sentiment classifier** — saving all results to a single output file.

---

## 📋 Overview

This project demonstrates a complete NLP workflow built with HuggingFace Transformers and TensorFlow/Keras:

1. **Translation** — Translates Spanish text to English using a MarianMT model
2. **Summarization** — Generates a concise English summary using BART
3. **Sentiment Analysis** — Classifies the sentiment using DistilBERT
4. **Custom LSTM Classifier** — Trains a lightweight LSTM model from scratch for 3-class sentiment prediction (positive / neutral / negative)
5. **Output Export** — Saves translation, summary, and sentiment results to a `.txt` file

---

## 🗂️ Project Structure

```
nlp-text-processing-pipeline/
│
├── nlp-text-processing.ipynb   # Main notebook
├── texto.txt                   # Input text in Spanish
├── resultados_nlp.txt          # Output file (generated after running)
└── README.md
```

---

## ⚙️ Requirements

- Python 3.10+
- PyTorch
- TensorFlow / Keras
- HuggingFace Transformers

Install dependencies:

```bash
pip install torch transformers tensorflow numpy pandas
pip install sacremoses  # recommended for MarianMT tokenizer
```

---

## 🚀 How to Run

1. Place your Spanish input text in `texto.txt`
2. Open `nlp-text-processing.ipynb` in Jupyter
3. Run all cells in order:

---

## 🤖 Models Used

| Model | Source | Task |
|-------|--------|------|
| `Helsinki-NLP/opus-mt-es-en` | HuggingFace | Spanish → English translation |
| `facebook/bart-large-cnn` | HuggingFace | Text summarization |
| `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace | Sentiment analysis |
| Custom LSTM | Built from scratch with Keras | 3-class sentiment (ES) |

---

## 🧱 LSTM Architecture

```
Embedding(vocab_size, 32, input_length=50)
    ↓
SpatialDropout1D(0.25)
    ↓
LSTM(50, dropout=0.5, recurrent_dropout=0.5)
    ↓
Dropout(0.2)
    ↓
Dense(3, activation='softmax')   # negative / neutral / positive
```

Compiled with:
- Loss: `sparse_categorical_crossentropy`
- Optimizer: `Adam`
- Metric: `accuracy`

> ⚠️ The custom LSTM is trained on a small dataset for educational purposes. For production-grade sentiment analysis, use the DistilBERT pipeline instead.

---

## 📌 Notes

- The translation is split into 500-character fragments to respect model input limits
- The LSTM is trained in Spanish; DistilBERT operates on the English translation
- `model.build(input_shape=(None, 50))` is required after `compile()` in newer Keras versions to display correct parameter counts

---

## 📚 References

- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [Helsinki-NLP MarianMT](https://huggingface.co/Helsinki-NLP/opus-mt-es-en)
- [facebook/bart-large-cnn](https://huggingface.co/facebook/bart-large-cnn)
- [DistilBERT SST-2](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english)
- [TensorFlow Keras](https://www.tensorflow.org/api_docs/python/tf/keras)
