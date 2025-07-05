
---

# Book Recommender System — LLM-Powered Semantic Discovery

A production-grade, **LLM-based book recommendation system** that uses deep language understanding, vector similarity, and emotional insight to deliver **deeply personalized and explainable** book suggestions.

Unlike traditional recommender systems that rely on ratings or collaborative filtering, this system leverages rich book metadata (e.g., descriptions), **OpenAI embeddings**, **LLM-powered classification**, and **emotion detection** to intelligently match, classify, and recommend books.

---

## Project Objective

To build a fully explainable, **language-first book discovery engine** that:

* Recommends books from **natural language queries** (e.g., *“Books about space exploration for children”*)
* Embeds book descriptions into vector space using **OpenAI Embeddings**
* Retrieves semantically similar books using **Chroma vector database**
* Classifies books as **Fiction**, **Nonfiction**, or **Children’s** using **zero-shot LLM classification**
* Analyzes the **emotional tone** of each book using a **fine-tuned emotion detection model**
* Powers an interactive **Gradio dashboard** for genre-, tone-, and meaning-based exploration

---

## 📁 Project Structure

```
book-recommender/
├── data-exploration.ipynb         # Data cleaning, wrangling, EDA
├── vector-search.ipynb            # Embeddings, vector store, semantic search
├── text-classification.ipynb      # Zero-shot LLM genre classification
├── sentiment-analysis.ipynb       # Emotion analysis using Hugging Face model
├── gradio-dashboard.py            # Interactive book recommender UI (Gradio)
├── tagged_descriptions.txt        # Text used for embedding (ISBN + description)
├── books_cleaned.csv              # Cleaned dataset post EDA
├── books_with_categories.csv      # Genre-classified dataset
├── books_with_emotions.csv        # Final enriched dataset (genre + emotions)
└── .env                           # Secure environment variables (API keys)
```

---

## Project Progress

### 1. Data Cleaning & Exploration [`data-exploration.ipynb`](./data-exploration.ipynb)

* Loaded the [7K Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset from Kaggle
* Filtered books with:

  * Complete descriptions (≥25 words)
  * Valid authors and titles
* Merged title and subtitle
* Engineered features:

  * `title_and_subtitle`, `age_of_book`, `tagged_description` (used for vector indexing)
* Output saved as `books_cleaned.csv`

---

### 🔍 2. Semantic Embedding & Vector Search [`vector-search.ipynb`](./vector-search.ipynb)

* Embedded book descriptions using OpenAI's [`text-embedding-ada-002`](https://platform.openai.com/docs/guides/embeddings)
* Split text into chunks using LangChain's `CharacterTextSplitter`
* Stored embeddings in **ChromaDB**
* Enabled vector-based semantic retrieval:

```
retrieve_semantic_recommendation("Books about nature for children")
```

* Mapped retrieved vectors to book metadata using `isbn13`

---

### 🧠 3. Genre Classification via Zero-Shot LLM [`text-classification.ipynb`](./text-classification.ipynb)

* Original dataset had **479+ inconsistent categories**
* Reduced to 3 major genres:

  * `Fiction`
  * `Nonfiction`
  * `Children's Nonfiction`
* Used [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli) with zero-shot classification
* Performance on validation set:

  * ✅ Accuracy: **77.8%**
  * ✅ F1 Scores: 0.75 (Fiction), 0.80 (Nonfiction)
* Classified remaining uncategorized books
* Output saved as `books_with_categories.csv`

---

### ❤️ 4. Emotion Detection for Tone Filtering [`sentiment-analysis.ipynb`](./sentiment-analysis.ipynb)

* Used [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) — a fine-tuned transformer model
* Target emotions:

  * `joy`, `sadness`, `fear`, `anger`, `disgust`, `surprise`, `neutral`
* Strategy:

  * Split each description into sentences
  * Run classifier on each sentence
  * Aggregate scores using **max pooling**
* Merged scores with original dataset
* Final dataset saved as `books_with_emotions.csv`

---

### 🖼️ 5. Interactive Book Recommender Dashboard [`gradio-dashboard.py`](./gradio-dashboard.py)

* Built a user-friendly, responsive dashboard using **Gradio**
* Allows users to:

  * Search books by natural language query
  * Filter by **genre** (Fiction, Nonfiction, etc.)
  * Filter by **emotional tone** (Happy, Sad, Suspenseful, etc.)
* Results include:

  * Book cover
  * Title + author
  * Short description preview
* Ready for deployment (next step)

---

## 🛠️ Tech Stack

| Component       | Technology                                                                                                              |
| --------------- | ----------------------------------------------------------------------------------------------------------------------- |
| Language Models | OpenAI Embeddings (`text-embedding-ada-002`), `facebook/bart-large-mnli`                                                |
| Emotion Model   | [`j-hartmann/emotion-english-distilroberta-base`](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base) |
| Frameworks      | LangChain, Hugging Face Transformers, Gradio                                                                            |
| Vector Storage  | ChromaDB (in-memory)                                                                                                    |
| Notebook Dev    | Jupyter (via PyCharm)                                                                                                   |
| Tooling         | pandas, tqdm, numpy, matplotlib, seaborn, dotenv, scikit-learn                                                          |

---

## 🚧 What’s Next?

🚀 **Coming up:**

* [x] Build Gradio dashboard for semantic recommendations
* [ ] 💻 **Deploy on Hugging Face Spaces** (free hosting for Gradio apps)
* [ ] 🧪 Evaluate recommender performance (e.g. NDCG, human evals)
* [ ] 🌐 Add public search/share features (e.g., public book URLs)
* [ ] 🎨 Design upgrade — improve UX/UI for visual appeal and mobile responsiveness
* [ ] 📊 Add explainability: show why each book was recommended (embedding similarity, emotional tone)

---

## 🧑‍💻 Contributing & License

This is a personal data science project exploring LLMs, vector search, and recommender systems.

* 📜 Dataset: [Kaggle Book Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
* 🤗 Models: Available via [Hugging Face](https://huggingface.co/)
* 🛠️ Open to suggestions and pull requests

---

Let me know when you're ready and I’ll help you:

* Clean and optimize your codebase
* Create `requirements.txt`, `README.md`, `app.py` for Hugging Face Spaces
* Add `.gitattributes` and secrets config

Ready when you are 🚀
