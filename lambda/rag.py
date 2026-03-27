import numpy as np

from api_clients import call_gemini_api, get_nomic_embedding
from config import GEMINI_API_KEY
from db import create_movie_documents


def perform_rag_query(query, faiss_index, conn):
    """
    Performs the full RAG workflow from embedding to LLM response.
    """
    if not GEMINI_API_KEY:
        return "Error: Gemini API key is not configured.", None

    try:
        query_vector = get_nomic_embedding(query, task_type="search_query")
        if query_vector is None:
            return "Error: Failed to get embedding from Nomic API.", None
    except ValueError as e:
        return f"Error: {e}", None

    query_vector = np.expand_dims(query_vector, axis=0)

    k = 5
    distances, indices = faiss_index.search(query_vector, k)
    sorted_indices = [x for _, x in sorted(zip(distances[0], indices[0]))]

    retrieved_docs = create_movie_documents(conn, movie_ids=sorted_indices)
    context = [doc["content"] for doc in retrieved_docs]
    context_str = "\n\n---\n\n".join(context)

    prompt = (
        "You are a helpful movie assistant.\\n\\n"
        "**The top {k} matching Documents are attached. They contain multiple pieces of information "
        "about a movie, including title, year, genre, director, cast, rating, and plot summary. "
        "Each document is separated by ---. Please provide a formatted non-markdown text response "
        "based on the context below. If the answer is not in the context, reply: 'Not enough info "
        "in IMDB top 250 to answer this'**\\n\\n"
        "**Context:**\\n---\\n{context_str}\\n---\\n\\n"
        "**User Question:** \\\"{query}\\\"\\n\\n"
        "**Answer:**"
    ).format(k=k, context_str=context_str, query=query)

    try:
        final_answer = call_gemini_api(prompt)
        if not final_answer:
            return "Error generating response from Gemini API.", None
        return final_answer, retrieved_docs
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        return f"Error communicating with LLM: {e}", None
