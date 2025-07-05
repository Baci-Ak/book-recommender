
# ========== Imports & Environment Setup ==========
import os
import numpy as np
import pandas as pd
import gradio as gr

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# ========== Load & Prepare Book Dataset ==========
books = pd.read_csv('books_with_emotions.csv')

# Add fallback thumbnail image for books with missing cover
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books['large_thumbnail'] = np.where(books['large_thumbnail'].isna(), 'no_cover.jpg', books['large_thumbnail'])

# ==========  Load Vector Database for Semantic Search ==========


embedding_model = OpenAIEmbeddings()
db_books = Chroma(persist_directory="chroma_store", embedding_function=embedding_model)


# ========== Semantic Retrieval Logic ==========
def retrieve_semantic_recommendations(
    query: str,
    category: str = None,
    tone: str = None,
    initial_top_k: int = 50,
    final_top_k: int = 24
) -> pd.DataFrame:
    # Search top-N semantically similar books
    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    # Apply category filter
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)

    else:
        book_recs = book_recs.head(final_top_k)

    # Apply emotion tone sorting
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        book_recs = book_recs.sort_values(by=tone_map[tone], ascending=False)

    return book_recs

# ========== Frontend Book Formatter ==========
def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Truncate description
        desc_words = row['description'].split()
        truncated_description = " ".join(desc_words[:30]) + "..."

        # Format authors nicely
        authors_split = row['authors'].split(';')
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])} and {authors_split[-1]}"
        else:
            authors_str = row['authors']

        # Create caption and append result
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results

# ========== Gradio UI ==========
# Dropdown choices
categories = ['All'] + sorted(books["simple_categories"].dropna().unique())
tones = ['All', 'Happy', 'Surprising', 'Angry', 'Suspenseful', 'Sad']

# Define Gradio Blocks Interface
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.HTML("<h1 style='font-size: 48px; color: #5A189A; text-align: center;'>Semantic Book Recommendation System</h1>")
    gr.Markdown("Enter your query and get LLM-powered book suggestions based on meaning, tone, and genre.")

    with gr.Row():
        user_query = gr.Textbox(label="Describe your book interest:", placeholder="e.g., A story about forgiveness and healing")
        category_dropdown = gr.Dropdown(choices=categories, label="Genre Filter:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Emotional Tone Filter:", value="All")
        submit_button = gr.Button("Find Recommendations")

    gr.Markdown("## Recommendations")
    output = gr.Gallery(label="Recommended Books", columns=8, rows=3)

    gr.HTML("""
        <div style='text-align: center; margin-top: 60px;'>
            <p style='font-size: 18px; color: #444;'><strong>Bassey Akom</strong></p>
            <p style='font-size: 16px; color: #666;'>Data Scientist / Mathematician</p>
            <a href='https://www.linkedin.com/in/basseyakom/' target='_blank' style='color: #5A189A; font-weight: bold;'>Connect on LinkedIn</a>
        </div>
    """)

    # Connect button to function
    submit_button.click(fn= recommend_books, inputs=[user_query, category_dropdown, tone_dropdown], outputs=output)

# Launch
if __name__ == '__main__':
    dashboard.launch()
