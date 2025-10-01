import psycopg2
import faiss
import numpy as np
import os
import nomic.embed as embed
import nomic.cli as nomic_login

# Ensure you're logged in to Nomic. Get the token from: https://atlas.nomic.ai/
nomic_login.login(token="<token>")


# --- Configuration ---
# Load credentials and settings from environment variables for security.
DB_NAME = "movies_pg_250"
DB_USER = "postgres"
DB_PASSWORD = "<pass>"
DB_HOST = "localhost"
EMBEDDING_MODEL = "nomic-embed-text:latest" # The Nomic embedding model.
VDB_PATH = "faiss_index3.bin"  # Path to save/load the FAISS index.

# --- 1. Database Interaction ---
def connect_to_postgres():
    """Establishes and returns a connection to the PostgreSQL database."""
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST
        )
        print("Successfully connected to PostgreSQL database.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def create_movie_documents(conn, movie_ids=None):
    """
    Fetches data from the SQL database and compiles it into
    comprehensive text documents for each movie.
    """
    print("4. Fetching movie data from local database and creating text documents...")
    cursor = conn.cursor()
    

    if movie_ids is None:
        cursor.execute("SELECT id FROM public.movies")
        movie_ids = [row[0] for row in cursor.fetchall()]

    movie_documents = []

    for movie_id in movie_ids:
        movie_id_str = str(movie_id)
        # Fetch basic movie info
        cursor.execute("SELECT title, year, kind, rating, votes, runtime, plot_summary FROM public.movies WHERE id = %s", (movie_id_str,))
        movie = cursor.fetchone()
        
        # Fetch genres
        cursor.execute("SELECT genre FROM public.genres WHERE movie_id = %s", (movie_id_str,))
        genres = ", ".join([row[0] for row in cursor.fetchall() if row[0]])
        
        # Fetch directors
        cursor.execute("""
            SELECT p.name FROM public.people p
            JOIN public.directors d ON p.id = d.person_id
            WHERE d.movie_id = %s
        """, (movie_id_str,))
        directors = ", ".join([row[0] for row in cursor.fetchall() if row[0]])
        
        # Fetch cast
        cursor.execute("""
            SELECT p.name, r.role FROM public.people p
            JOIN public.roles r ON p.id = r.person_id
            WHERE r.movie_id = %s
        """, (movie_id_str,))
        cast_list = [f"{row[0]} as {row[1]}" for row in cursor.fetchall() if row[0] and row[1]]
        cast = ", ".join(cast_list)

        # Assemble the document
        doc_text = f"""Title: {movie[0]}
                        Year: {movie[1]}
                        Type: {movie[2]}
                        Rating: {movie[3]}/10 from {movie[4]} votes.
                        Runtime: {movie[5]} minutes
                        Genre(s): {genres}
                        Directed by: {directors}
                        Cast: {cast}
                        Plot Summary: {movie[6]}"""
        
        movie_documents.append({
            "id": movie_id_str,
            "content": doc_text
        })

    return movie_documents

# --- 2. Embedding and Vector Database ---
def create_and_populate_vector_db(documents):
    """
    Generates embeddings using a local Ollama model and stores them in a FAISS index.
    """
    contents = [doc['content'] for doc in documents]
    movie_ids = np.array([doc['id'] for doc in documents])
    
    print(f"Generating embeddings for {len(contents)} documents with '{EMBEDDING_MODEL}'...")
    try:
        # Generate embeddings for each document
        embeddings_array = embed.text([content for content in contents], task_type='search_document', model=EMBEDDING_MODEL)['embeddings']
    except Exception as e:
        print(f"Error communicating with Nomic AI for embeddings: {e}")
        return None, None

    embeddings = np.array([row for row in embeddings_array]).astype('float32')
    
    embedding_dimension = embeddings.shape[1]
    print(f"Embeddings generated with dimension: {embedding_dimension}")
    
    # Create a FAISS index
    index = faiss.IndexFlatL2(embedding_dimension)
    index = faiss.IndexIDMap(index) # Allows mapping vectors to original doc IDs
    # Add vectors to the index
    index.add_with_ids(embeddings, movie_ids)
    
    print(f"Successfully populated FAISS index with {index.ntotal} vectors.")
    return index


# --- Main Execution ---
if __name__ == '__main__':

    if VDB_PATH in os.listdir():
        pass
    # Validate environment setup
    elif not all([DB_NAME, DB_USER, DB_PASSWORD]):
        print("CRITICAL: One or more environment variables (DB_NAME, DB_USER, DB_PASSWORD) are not set.")
    else:
        db_connection = connect_to_postgres()
        if db_connection:
            # 1. Prepare data
            movie_docs = create_movie_documents(db_connection)
            
            if not movie_docs:
                 print("No movie documents were created. Please check if your database tables are populated.")
            else:
                # 2. Create Vector Store
                faiss_index = create_and_populate_vector_db(movie_docs)
                #store the faiss index

                faiss.write_index(faiss_index, VDB_PATH)

            db_connection.close()
            print("Database connection closed.")

# --- Optional: Create a Word Cloud for Visualization ---
def create_wordcloud():

    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    db_connection = connect_to_postgres()
    if db_connection:
        movie_docs = create_movie_documents(db_connection)
        db_connection.close()
        print("Database connection closed.")
    
        text = " ".join([doc['content'] for doc in movie_docs])
        
        #clean up the text by removing non alphanumeric characters
        import re
        text = re.sub(r'\W+', ' ', text)
        #filter stop words
        stop_words = set(["the", "and", "of", "to", "a", "in", "is", "it", "that", "as", "with", "for", "its", "on", "by", "an", "at", "from", "this", "be", "are", "was", "but", "not", "or", "have", "has", "they", "you", "his", "her", "he", "she", "type", "s", "quot", "href", "movie", "md"])
        text = " ".join([word for word in text.split() if word.lower() not in stop_words])
        # Generate a word cloud image
        wordcloud = WordCloud(width=2000, height=1000, margin=0, collocations=False, relative_scaling=0).generate(text)
        #save the generated image
        wordcloud.to_file("movie_wordcloud.png")
        print("Word cloud image saved as 'movie_wordcloud.png'")
        # Display the generated image:
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()