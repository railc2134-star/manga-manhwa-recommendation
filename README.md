# IntentRouterAI

A lightweight NLP system that classifies user messages into intent categories using **Jina embeddings** and a neural network classifier.

The system converts raw text into semantic embeddings and predicts the intent class.

---

## 🧠 Problem Solved

Given a message, classify it into:

- **0 → Manga / Manhwa / Content Discovery**
- **1 → Social / Chat / Casual conversation**
- **2 → System / Control / Commands**

---

## ⚙️ Architecture

The pipeline is simple but effective:

### 1. Text Embedding
- Uses Jina embeddings API (`jina-embeddings-v2-base-en`)
- Converts text → 768-dimensional vector

### 2. Classifier Model
A small neural network:
