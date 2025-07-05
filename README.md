---


# 📚 Book Recommender System — LLM-Powered Semantic Discovery

A production-grade, **LLM-based book recommendation system** that uses deep language understanding and vector similarity search to deliver highly relevant and personalized book suggestions.

Unlike traditional recommender systems that rely on user ratings or collaborative filtering, this system leverages rich textual metadata (e.g. descriptions) and **OpenAI embeddings** to perform intelligent matching and classification of books based on **semantic meaning**.

---

## 🎯 Project Objective

To build a **fully explainable, language-first book recommendation engine** that:

- Recommends books based on natural language queries (e.g. *“Books about space exploration for children”*)
- Embeds book metadata into a vector space using LLM embeddings
- Enables semantic search through vector similarity (using `Chroma`)
- Classifies books by **Fiction vs Nonfiction** using **zero-shot LLM classification**
- Lays the foundation for **filterable, personalized discovery**

---

## 📁 Project Structure

```

book-recommender/
├── data-exploration.ipynb       # End-to-end data cleaning, EDA, wrangling
├── vector-search.ipynb          # Embedding, vector store, semantic querying
├── text-classification.ipynb    # LLM-based category classification (fiction/nonfiction)
├── books_cleaned.csv            # Cleaned metadata used for embeddings
├── books_with_categories.csv    # Final dataset with category classification
└── .env                         # Environment variables (e.g. OpenAI API key)

````

---

## ✅ Project Progress

### 📊 1. Data Cleaning & Exploration (`data-exploration.ipynb`)
- ✅ Loaded the **7K Books with Metadata** dataset from Kaggle [Kaggle: 7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
- ✅ Conducted **EDA** to identify and handle missing values
- ✅ Created `tagged_description` (ISBN + description) for indexing
- ✅ Filtered books with at least **25+ words** in the description
- ✅ Merged subtitle with title when available
- ✅ Engineered fields like:
  - `title_and_subtitle`
  - `tagged_description`
  - `age_of_book`
- ✅ Saved cleaned dataset to `books_cleaned.csv`

---

### 🔎 2. Semantic Embedding & Vector Search (`vector-search.ipynb`)
- ✅ Extracted book descriptions for embedding
- ✅ Chunked long text using `CharacterTextSplitter`
- ✅ Used **OpenAI's Embeddings API** via `langchain` to convert books into vector space
- ✅ Stored vectorized data in **ChromaDB** for fast retrieval
- ✅ Implemented semantic query interface:

```
  retrieve_semantic_recommendation("A book about nature for kids")
````

✅ Created utility function to fetch metadata based on `isbn13`

---

### 🧠 3. LLM-Based Text Classification (`text-classification.ipynb`)

* ✅ Cleaned and simplified `categories` field (originally 470+ messy values)
* ✅ Created `simple_categories` with values: `Fiction`, `Nonfiction`, and `Children's`
* ✅ Used `facebook/bart-large-mnli` via HuggingFace to perform **zero-shot classification**
* ✅ Evaluated performance:

  * **Accuracy**: 77.8%
  * **F1 Scores**: `0.75` (Fiction), `0.80` (Nonfiction)
* ✅ Applied predictions to books with missing categories
* ✅ Saved updated dataset as `books_with_categories.csv`

---

## 🛠️ Tech Stack

| Component      | Technology                                       |
| -------------- | ------------------------------------------------ |
| Language Model | OpenAI Embeddings (`text-embedding-ada-002`)     |
| Frameworks     | LangChain, Hugging Face Transformers             |
| Data Storage   | ChromaDB (in-memory vector store)                |
| Notebook Dev   | Jupyter (inside PyCharm)                         |
| Libraries      | pandas, matplotlib, seaborn, transformers, numpy |
| Environment    | `.venv`, `.env`, macOS M1 (MPS acceleration)     |

---

## 🚧 What's Next?

Coming up in the next stages:

* 📈 **Sentiment Analysis** on book descriptions
* 🎛️ **Category-based filtering** in recommendations
* 🖼️ Web UI / Streamlit frontend
* 🔌 API deployment (FastAPI or LangServe)
* 🧪 Evaluation using user study / embeddings comparison

---

## 🤝 Contributing & License

This is a personal project. Contributions and suggestions are welcome as the project evolves into a more robust recommender platform.

---

