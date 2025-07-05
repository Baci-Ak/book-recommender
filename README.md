Here is the updated **world-class `README.md`** for your project, including the new **Sentiment Analysis** stage. This documentation is fully structured, professional, and follows industry best practices:

---

# 📚 Book Recommender System — LLM-Powered Semantic Discovery

A production-grade, **LLM-based book recommendation system** that uses deep language understanding and vector similarity search to deliver highly relevant and personalized book suggestions.

Unlike traditional recommender systems that rely on user ratings or collaborative filtering, this system leverages rich textual metadata (e.g. descriptions), **OpenAI embeddings**, and **LLM-powered classification** to perform intelligent book matching, classification, and emotional analysis.

---

## 🎯 Project Objective

To build a fully explainable, **language-first book recommendation engine** that:

* Recommends books from natural language queries (e.g. *“Books about space exploration for children”*)
* Embeds book descriptions into a vector space using **OpenAI Embeddings**
* Retrieves semantically similar content using **Chroma vector search**
* Classifies books into **Fiction**, **Nonfiction**, or **Children’s** categories using **zero-shot LLM classification**
* Analyzes the **emotional tone** of each book using a **fine-tuned emotion classifier**
* Lays the foundation for **filterable, emotion-aware, personalized recommendations**

---

## 📁 Project Structure

```
book-recommender/
├── data-exploration.ipynb         # Data cleaning, wrangling, and EDA
├── vector-search.ipynb            # Embeddings, vector store, semantic queries
├── text-classification.ipynb      # Zero-shot LLM-based genre classification
├── sentiment-analysis.ipynb       # Emotion classification using fine-tuned model
├── books_cleaned.csv              # Dataset after filtering and preprocessing
├── books_with_categories.csv      # Dataset after genre classification
├── books_with_emotions.csv        # Final dataset with emotion scores
└── .env                           # API keys (e.g. OpenAI) and environment vars
```

---

## ✅ Project Progress

### 📊 1. Data Cleaning & Exploration [`data-exploration.ipynb`](./data-exploration.ipynb)

* Loaded the [7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset from Kaggle
* Cleaned and filtered:

  * Removed rows missing critical metadata
  * Kept books with **25+ word descriptions**
  * Combined `title` and `subtitle` intelligently
* Engineered features:

  * `tagged_description` = `isbn13 + description`
  * `title_and_subtitle`, `age_of_book`, etc.
* Saved cleaned data as `books_cleaned.csv`

---

### 🔍 2. Semantic Embedding & Vector Search [`vector-search.ipynb`](./vector-search.ipynb)

* Embedded descriptions using OpenAI's [`text-embedding-ada-002`](https://platform.openai.com/docs/guides/embeddings)
* Split text into chunks using `CharacterTextSplitter`
* Stored vectors using **ChromaDB**
* Implemented semantic recommendation:

  ```
  retrieve_semantic_recommendation("A book to teach children about nature")
  ```
* Mapped recommendations back to metadata via `isbn13`

---

### 🧠 3. LLM-Based Genre Classification [`text-classification.ipynb`](./text-classification.ipynb)

* Original `categories` field had **479 messy labels**
* Consolidated into 3 clean classes:

  * `Fiction`
  * `Nonfiction`
  * `Children's Nonfiction`
* Used `facebook/bart-large-mnli` for **zero-shot classification** via Hugging Face
* Evaluated on 600 samples (300 per class):

  * **Accuracy**: 77.8%
  * **F1 Score**: 0.75 (Fiction), 0.80 (Nonfiction)
* Applied model to classify books with missing categories
* Saved as `books_with_categories.csv`

---

### ❤️ 4. Emotion Classification with Fine-Tuned LLM [`sentiment-analysis.ipynb`](./sentiment-analysis.ipynb)

* Used `j-hartmann/emotion-english-distilroberta-base` fine-tuned model from [Hugging Face](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base)
* Targeted **7 emotions**:

  * `joy`, `sadness`, `fear`, `anger`, `disgust`, `surprise`, `neutral`
* Strategy:

  * Split each book description into sentences
  * Applied classifier per sentence
  * Aggregated scores using **max pooling**
* Merged predicted emotion scores with original book metadata
* Final dataset saved as `books_with_emotions.csv`

---

## 🛠️ Tech Stack

| Component       | Technology                                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Language Models | OpenAI Embeddings (`text-embedding-ada-002`), `facebook/bart-large-mnli`                                                |
| Emotion Model   | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) |
| Frameworks      | LangChain, Hugging Face Transformers                                                                                    |
| Vector Storage  | ChromaDB (in-memory vector database)                                                                                    |
| Dev Environment | Jupyter via PyCharm, macOS M1 with MPS acceleration                                                                     |
| Tooling         | pandas, tqdm, numpy, matplotlib, seaborn, scikit-learn, dotenv                                                          |

---

## 🚧 What’s Next?

The next stages in the roadmap:

* 🖼️ Build interactive UI with **Gradio**
* 🎛️ Add **filters** based on genre or emotion
* 🔌 Deploy as API (FastAPI / LangServe)
* 🧪 Conduct evaluation and user studies
* 🔎 Explainability via embeddings visualization

---

## 🤝 Contributing & License

This is a personal project exploring the fusion of NLP and recommender systems. Contributions, ideas, or issue reports are welcome.

* 🧠 Model Licensing: All models used are publicly available on [Hugging Face](https://huggingface.co/)
* 📜 Dataset License: [Kaggle Book Metadata Dataset](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)

---


