# 📚 Book Recommender System (LLM-Powered)

A production-ready book recommendation engine powered by large language models (LLMs) and semantic vector search. The system uses rich metadata and natural language embeddings to deliver content-aware book recommendations beyond traditional ratings or genres.

---

## 🧠 Project Goal

To build a **semantic recommender system** that understands book content using LLM embeddings (e.g. `text-embedding-ada-002` from OpenAI). This allows recommendations based on **meaning**, not just popularity or similarity in metadata.

---

## ✅ Project Progress

### 📊 1. Data Exploration & Wrangling
- Loaded high-quality dataset from [Kaggle: 7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
- Cleaned and filtered:
  - Retained only books with complete and meaningful descriptions
  - Merged titles and subtitles where available
  - Engineered fields for embedding, indexing, and similarity search
- Saved cleaned data as `books_cleaned.csv`
- All preprocessing steps are reproducible and documented in `data-exploration.ipynb`
