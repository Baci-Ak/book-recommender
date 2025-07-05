Perfect — you're now ready to update your `README.md` to match your current Hugging Face-compatible, **production-grade setup**.

Here’s a polished and **world-class** version of your `README.md`, reflecting your updated `tree`, renamed files (`app.py`), actual folder structure (`chroma_store/`), and production-ready functionality:

---

```markdown
---
title: Semantic Book Recommender
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.35.0
app_file: app.py
pinned: false
---

# 📚 Semantic Book Recommender — LLM-Powered Discovery

A **next-gen book recommendation system** that uses LLMs, OpenAI embeddings, and emotion detection to help you find books not just by genre, but by **meaning** and **emotional tone**.

This project is an end-to-end pipeline — from data wrangling to vector storage to an interactive Gradio UI — for **intelligent book recommendations** based on *how a book feels* and *what it's about*, not just what category it's in.

---

## 🔍 Key Features

✅ Search by **natural language** (e.g. *“inspiring memoirs about overcoming loss”*)  
✅ Filter by **genre** (fiction, nonfiction, children’s)  
✅ Filter by **emotion** (joy, sadness, fear, etc.)  
✅ Powered by **OpenAI embeddings + Chroma vector DB**  
✅ Interactive UI via **Gradio**  
✅ 100% local inference — fast and free

---

## 🗂️ Project Structure

```

book-recommender/
├── app.py                         # Main app for Hugging Face deployment
├── gradio-dasboard-main.py       # Older version (local dev)
├── build\_vector\_store.py         # One-time script to build Chroma vector DB
├── chroma\_store/                 # Vector DB folder (auto-created)
│   └── ...
├── tagged\_descriptions.txt       # ISBN + descriptions for embedding
├── no\_cover.jpg                  # Placeholder image
├── requirements.txt              # Project dependencies
├── README.md                     # You're here
├── books\_cleaned.csv             # Cleaned metadata
├── books\_with\_categories.csv     # Genre-classified dataset
├── books\_with\_emotions.csv       # Final enriched dataset (used in app)
├── data-exploration.ipynb        # EDA, cleaning
├── vector-search.ipynb           # Embeddings + Chroma indexing
├── text-classification.ipynb     # Zero-shot genre classification
└── sentiment-analysis.ipynb      # Emotion tagging using Hugging Face

````

---

## ⚙️ How It Works

### 🧹 1. Data Preparation

We clean and filter a Kaggle dataset of 7,000+ books:

- Remove books with short/no descriptions
- Merge title and subtitle
- Engineer `tagged_description` (ISBN + description) for embedding

📄 Output: `books_cleaned.csv`

---

### 🧠 2. Semantic Embeddings + Vector Search

We encode book descriptions using **OpenAI’s `text-embedding-ada-002`**, chunk them, and index them into a **ChromaDB vector store** for fast semantic similarity search.

📄 Script: `vector-search.ipynb`  
🧠 Vector DB: `chroma_store/`

---

### 🏷️ 3. Genre Classification (Zero-Shot LLM)

Using [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli), we classify each book into:

- `Fiction`
- `Nonfiction`
- `Children's Nonfiction`

No labels were needed — it's **zero-shot**.

📄 Output: `books_with_categories.csv`

---

### ❤️ 4. Emotion Detection

Using [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base), we detect emotions like:

- joy
- sadness
- fear
- surprise
- anger
- disgust

This allows users to **filter books by emotional tone**, e.g., *“surprising nonfiction”* or *“joyful stories for kids”*.

📄 Output: `books_with_emotions.csv`

---

### 💡 5. Semantic Recommender UI

`app.py` launches a **Gradio dashboard** that lets users:

- 🔍 Enter a natural language query
- 🎨 Filter by **genre**
- ❤️ Filter by **emotional tone**
- 🖼️ See covers, authors, and short descriptions

Live on [Hugging Face Spaces](https://huggingface.co/spaces)!

---

## 🧰 Tech Stack

| Component       | Technology                                                                                                              |
|----------------|--------------------------------------------------------------------------------------------------------------------------|
| Embeddings     | OpenAI `text-embedding-ada-002`                                                                                          |
| Genre Classifier | `facebook/bart-large-mnli` (zero-shot classification)                                                                  |
| Emotion Model  | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) |
| Vector Search  | ChromaDB                                                                                                                 |
| UI Framework   | Gradio                                                                                                                   |
| NLP Pipeline   | LangChain                                                                                                                |
| Dev Tools      | pandas, numpy, tqdm, dotenv                                                                                              |

---

## 🚀 Run Locally

```bash
# 1. Clone repo
git clone https://github.com/your-username/book-recommender.git
cd book-recommender

# 2. Set API key
echo "OPENAI_API_KEY=your-key" > .env

# 3. Install deps
pip install -r requirements.txt

# 4. Launch app
python app.py
````

---

## ✅ Status & Roadmap

* [x] Semantic search via OpenAI embeddings
* [x] Genre classification via zero-shot LLM
* [x] Emotion filtering with transformer models
* [x] Gradio dashboard (v1)
* [x] Deployment to Hugging Face Spaces
* [ ] Public result sharing
* [ ] Better mobile UX
* [ ] Human evaluations (NDCG, recall)
* [ ] Embedding explainability in UI

---

## 📘 Credits

* 📚 [7K Books Metadata Dataset](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
* 🤗 Hugging Face Transformers
* 🧠 OpenAI API
* 🛠️ LangChain + ChromaDB
* 🎨 Gradio

---

## 👋 Contributing

Suggestions, PRs, and feedback are welcome! This is a personal learning project on LLMs, emotion-aware systems, and recommendation pipelines.

---

## 📄 License

MIT License. Open-source and open to the world 🌍

```

---

Let me know if you'd like me to generate this directly as `README.md` and upload it for you — or if you want a version tailored for GitHub *and* Hugging Face at once.
```
