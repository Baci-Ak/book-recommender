
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

# Book Recommender System — LLM-Powered Semantic Discovery

A **book recommendation system** that uses LLM, OpenAI Embeddings, and langchain to deliver **personalized and explainable** book recommendation to target users.

Unlike traditional recommender systems that rely on ratings or collaborative filtering, this system leverages book metadata (descriptions), **OpenAI embeddings**, **LLM-powered classification**, and **emotion detection** to intelligently match, classify, and recommend books to target users.


---

## Business Context and Problem

With the ever-growing online content and the ubiquity of Internet-enabled devices, users are facing information overload. This overload leads to difficulties in making decisions on what content or products to consume or purchase. The project focuses on helping an online book selling company that has been experiencing declining sales due to the overwhelming choices and challenges customers face when selecting books to buy.

The company's management recognizes the need for an intelligent recommendation engine to assist users in finding books aligned with their emotions, meaning, and interest. This engine will not only enhance sales by providing personalized recommendations but also improve customer satisfaction by simplifying the decision-making process.

---
## Project Objective

To build a fully explainable, **language book recommender engine** that:

* Recommends books from **natural language queries** (e.g., *“Books about space exploration for children”*)
* Embeds book descriptions into vector space using **OpenAI Embeddings**
* Retrieves semantically similar books using **Chroma vector database**
* Classifies books as **Fiction**, **Nonfiction**, or **Children’s** using **zero-shot LLM classification**
* Analyzes the **emotional tone** of each book using a **fine-tuned emotion detection model**
* Powers an interactive **Gradio dashboard** for genre-, tone-, and meaning-based exploration

---

##  Project Structure

```
book-recommender/
├── app.py # Main app for Hugging Face deployment
├── gradio-dasboard-main.py # Older version (local dev)
├── build_vector_store.py # One-time script to build Chroma vector DB
├── chroma_store/ # Vector DB folder (auto-created)
│ └── ...
├── tagged_descriptions.txt # ISBN + descriptions for embedding
├── no_cover.jpg # Placeholder image
├── requirements.txt # Project dependencies
├── README.md # doc
├── books_cleaned.csv # Cleaned metadata
├── books_with_categories.csv # Genre-classified dataset
├── books_with_emotions.csv # Final enriched dataset (used in app)
├── data-exploration.ipynb # EDA, cleaning
├── vector-search.ipynb # Embeddings + Chroma indexing
├── text-classification.ipynb # Zero-shot genre classification
└── sentiment-analysis.ipynb # Emotion tagging using Hugging Face
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

  *  Accuracy: **77.8%**
  * F1 Scores: 0.75 (Fiction), 0.80 (Nonfiction)
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

[ Live App >>](https://baciakom-semantic-book-recommender.hf.space/)
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

## Run Locally?

---
```
# 1. Clone repo
git clone https://github.com/Baci-Ak/book-recommender.git
cd book-recommender

# 2. Set API key
echo "OPENAI_API_KEY=your-key" > .env

# 3. Install deps
pip install -r requirements.txt

# 4. Launch app
python app.py
```
---




## Status & Roadmap

* [x] Semantic search via OpenAI embeddings
* [x] Genre classification via zero-shot LLM
* [x] Emotion filtering with transformer models
* [x] Gradio dashboard (v1)
* [x] Deployment to Hugging Face Spaces
* [x] Public result sharing

---

## 🧑‍💻 Contributing & License

* 📜 Dataset: [Kaggle Book Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
* 🤗 Models: Available via [Hugging Face](https://huggingface.co/)
* 🛠️ Open to suggestions and pull requests

---
