"""Modulo: Dashboard"""

import os
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import gradio as gr

load_dotenv()

books = pd.read_csv("./datasets/4_books_with_emotion_scores.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "./src/public/img/image-not-found.png",
    books["large_thumbnail"],
)

# C:\Programacion\0) Practicas\Python - ML\Semantic Book Recommender\datasets\tagged_description.txt
# raw_documents = TextLoader("./datasets/tagged_description.txt").load()
PERSIST_DIR = "./datasets/vector_db"  # Asegúrate que sea un path relativo válido

embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Crear el directorio si no existe
if not os.path.exists(PERSIST_DIR):
    # Verifica la ruta del archivo
    file_path = os.path.abspath("./datasets/tagged_description.txt")
    print(f"Ruta completa intentada: {file_path}")  # <-- Imprime la ruta real

    # Validación del archivo
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"El archivo {file_path} no existe.")

    print("Creando la base de datos vectorial...")

    # Cargar documentos (ahora se usará la ruta absoluta)
    raw_documents = TextLoader(file_path, encoding="utf-8").load()

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    db_books = Chroma.from_documents(
        documents,
        embedding=embeddings_model,
        persist_directory=PERSIST_DIR
    )
else:
    print("Cargando la base de datos existente...")
    db_books = Chroma(
        embedding_function=embeddings_model,
        persist_directory=PERSIST_DIR
    )

# embeddings_model = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2"
# )
# # Cargar la base de datos desde el directorio persistente
# db_books = Chroma(
#     persist_directory="../datasets/chroma_db",
#     embedding_function=embeddings_model
# )

def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    """xxx"""

    recs = db_books.similarity_search(query, k=initial_top_k)
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(
        query: str,
        category: str,
        tone: str
):
    """xxx"""

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    print(f"# de resultados: {len(results)}")

    return results

categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme = gr.themes.Ocean()) as dashboard:
    gr.Markdown("# Find your perfect book")

    with gr.Row():
        user_query = gr.Textbox(
            label = "Description of the book:",
            placeholder = "A romantic story..."
        )

        category_dropdown = gr.Dropdown(
            choices = categories,
            label = "Select a category:",
            value = "All"
        )

        tone_dropdown = gr.Dropdown(
            choices = tones,
            label = "Select an emotional tone:",
            value = "All"
        )

        submit_button = gr.Button("Search")

    gr.Markdown("## List of books")

    output = gr.Gallery(
        # label = "Recommended books",
        columns = 8,
        rows = 2
    )

    submit_button.click(
        fn = recommend_books,
        inputs = [user_query, category_dropdown, tone_dropdown],
        outputs = output
    )

    gr.HTML(
        """
        <div style="text-align: center; padding: 20px; position: fixed; bottom: 2rem; width: 100%; left: 0;">
            <p>A Machine Learning project with Data Analysis, Vector Search, Text Classification and Sentiment Analysis</p>
            <p>Developed by <a href="https://ivin-dev.com/#about" target="_blank">Ivin Dev</a></p>
        </div>
        """
    )


if __name__ == "__main__":
    # Se especifica la IP y Puerto para que no haya temas en el deploy
    # Tiene que ser con "0.0.0.0", no con "127.0.0.1" para que funcione con Docker
    dashboard.launch(server_name="0.0.0.0", server_port=7860)
    # dashboard.launch()
