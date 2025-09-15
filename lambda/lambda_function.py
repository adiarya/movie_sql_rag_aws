import json
import os
import boto3
import faiss
import numpy as np
import requests
import psycopg2

# --- Configuration ---
# Read from environment variables for security and flexibility
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
S3_BUCKET = os.environ.get("S3_BUCKET")
FAISS_KEY = os.environ.get("FAISS_KEY")
FAISS_PATH = '/tmp/faiss_index3.bin'
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NOMIC_API_KEY = os.environ.get("NOMIC_API_KEY")

# Initialize AWS S3 client outside the handler for Lambda warm starts
s3_client = boto3.client('s3')

# Global variable to cache the FAISS index in memory for performance
cached_faiss_index = None

# --- Gemini API Helper ---
def call_gemini_api(prompt):
    """
    Calls the Gemini API for text generation using the provided prompt.
    Returns the generated text or None on error.
    """
    if GEMINI_API_KEY is None:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }]
    }
    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        # Extract the generated text from the response
        return result['candidates'][0]['content']['parts'][0]['text'].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return None

# --- Nomic API Embedding ---
def get_nomic_embedding(text, task_type='search_query'):
    """
    Generates an embedding for a given text using the Nomic API.
    Returns a numpy array or None on error.
    """
    if NOMIC_API_KEY is None:
        raise ValueError("NOMIC_API_KEY environment variable is not set.")

    api_url = "https://api-atlas.nomic.ai/v1/embedding/text"
    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "texts": [text],
        "model": "nomic-embed-text-v1.5",
        "task_type": task_type
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        embedding_data = response.json()
        return np.array(embedding_data['embeddings'][0]).astype('float32')
    except requests.exceptions.RequestException as e:
        print(f"Error calling Nomic API: {e}")
        return None

# --- FAISS Index Loading ---
def load_faiss_index_from_s3():
    """
    Downloads the FAISS index from S3 to the Lambda /tmp directory and loads it.
    Uses a global cache to avoid repeated downloads in warm Lambda invocations.
    """
    global cached_faiss_index
    if cached_faiss_index is None:
        if not all([S3_BUCKET, FAISS_KEY]):
            raise ValueError("S3_BUCKET and FAISS_KEY environment variables must be set.")
        try:
            print(f"Downloading FAISS index from s3://{S3_BUCKET}/{FAISS_KEY}...")
            s3_client.download_file(S3_BUCKET, FAISS_KEY, FAISS_PATH)
            cached_faiss_index = faiss.read_index(FAISS_PATH)
            print("FAISS index loaded successfully.")
        except Exception as e:
            print(f"Error loading FAISS index: {e}")
            return None
    return cached_faiss_index

# --- Database Functions ---
def connect_to_postgres():
    """
    Establishes a connection to the PostgreSQL database.
    Returns a connection object or None on failure.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=5432
        )
        print("Successfully connected to PostgreSQL database.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def create_movie_documents(conn, movie_ids):
    """
    Retrieves movie information from the database for the given IDs efficiently.
    Uses batch queries to minimize database round-trips.
    Returns a list of movie document dicts.
    """
    cursor = conn.cursor()
    movie_documents = []

    if not movie_ids:
        return movie_documents

    # Convert movie_ids to a list of strings and pad with leading zeros
    padded_movie_ids = [str(mid).zfill(7) for mid in movie_ids]
    id_list_sql = ', '.join([f"'{mid}'" for mid in padded_movie_ids])

    # 1. Fetch all movie details in a single query
    cursor.execute(f"SELECT id, title, year, kind, rating, votes, runtime, plot_summary FROM public.movies WHERE id IN ({id_list_sql})")
    movie_details = {row[0]: row[1:] for row in cursor.fetchall()}

    # 2. Fetch all genres in a single query
    cursor.execute(f"SELECT movie_id, genre FROM public.genres WHERE movie_id IN ({id_list_sql})")
    genres_map = {}
    for movie_id, genre in cursor.fetchall():
        genres_map.setdefault(movie_id, []).append(genre)

    # 3. Fetch all directors in a single query
    cursor.execute(f"""
        SELECT d.movie_id, p.name FROM public.people p
        JOIN public.directors d ON p.id = d.person_id
        WHERE d.movie_id IN ({id_list_sql})
    """)
    directors_map = {}
    for movie_id, name in cursor.fetchall():
        directors_map.setdefault(movie_id, []).append(name)
    
    # 4. Fetch all cast roles in a single query
    cursor.execute(f"""
        SELECT r.movie_id, p.name, r.role FROM public.people p
        JOIN public.roles r ON p.id = r.person_id
        WHERE r.movie_id IN ({id_list_sql})
    """)
    cast_map = {}
    for movie_id, name, role in cursor.fetchall():
        cast_map.setdefault(movie_id, []).append(f"{name} as {role}")

    # Build the document for each movie
    for movie_id in padded_movie_ids:
        if movie_id not in movie_details:
            continue

        title, year, kind, rating, votes, runtime, plot_summary = movie_details[movie_id]
        genres = ", ".join(genres_map.get(movie_id, []))
        directors = ", ".join(directors_map.get(movie_id, []))
        cast = ", ".join(cast_map.get(movie_id, []))

        doc_text = (
            f"Title: {title}\nYear: {year}\nType: {kind}\nRating: {rating}/10 from {votes} votes.\n"
            f"Runtime: {runtime} minutes\nGenre(s): {genres}\nDirected by: {directors}\nCast: {cast}\nPlot Summary: {plot_summary}"
        )
        movie_documents.append({"id": movie_id, "content": doc_text})

    cursor.close()
    return movie_documents

# --- RAG Workflow ---
def perform_rag_query(query, faiss_index, conn):
    """
    Performs the full Retrieval-Augmented Generation (RAG) workflow:
    1. Embeds the user query.
    2. Searches the FAISS index for similar movies.
    3. Retrieves movie details from the database.
    4. Calls Gemini LLM to generate a final answer.
    Returns (answer, context_docs).
    """
    if not GEMINI_API_KEY:
        return "Error: Gemini API key is not configured.", None
    
    try:
        query_vector = get_nomic_embedding(query, task_type='search_query')
        if query_vector is None:
            return "Error: Failed to get embedding from Nomic API.", None
    except ValueError as e:
        return f"Error: {e}", None

    query_vector = np.expand_dims(query_vector, axis=0)
    
    k = 5  # Number of documents to retrieve
    distances, indices = faiss_index.search(query_vector, k)
    # Sort indices by distance (optional, as FAISS returns sorted by default)
    sorted_indices = [x for _, x in sorted(zip(distances[0], indices[0]))]
    
    retrieved_docs = create_movie_documents(conn, movie_ids=sorted_indices)
    context = [doc['content'] for doc in retrieved_docs]
    context_str = "\n\n---\n\n".join(context)

    prompt = (
        f"You are a helpful movie assistant.\n\n"
        f"**The top {k} matching Documents are attached. They contain multiple pieces of information about a movie, including title, year, genre, director, cast, rating, and plot summary. Each document is separated by ---. Please provide a response based on the context below.**\n\n"
        f"**Context:**\n---\n{context_str}\n---\n\n"
        f"**User Question:** \"{query}\"\n\n"
        f"**Answer:**"
    )

    try:
        final_answer = call_gemini_api(prompt)
        if not final_answer:
            return "Error generating response from Gemini API.", None
        return final_answer, retrieved_docs
    except Exception as e:
        print(f"Error communicating with LLM: {e}")
        return f"Error communicating with LLM: {e}", None

# --- NL to SQL and Query Functions ---
def convert_nl_to_sql(user_query):
    """
    Converts a natural language query to a SQL query using Gemini LLM.
    Returns the SQL query string or None on error.
    """
    if not GEMINI_API_KEY:
        return None
        
    # Simplified schema for the LLM
    schema = """
    -- Tables: Movies, People, Directors, Roles, Genres
    CREATE TABLE Movies (
        id TEXT PRIMARY KEY,
        title TEXT,
        year INTEGER,
        kind TEXT,
        rating REAL,
        votes INTEGER,
        runtime INTEGER,
        plot_summary TEXT
    );
    CREATE TABLE People ( id TEXT PRIMARY KEY, name TEXT );
    CREATE TABLE Directors ( movie_id TEXT, person_id TEXT );
    CREATE TABLE Roles ( movie_id TEXT, person_id TEXT, role TEXT );
    CREATE TABLE Genres ( movie_id TEXT, genre TEXT );

    -- Foreign Keys:
    -- Directors.movie_id -> Movies.id
    -- Directors.person_id -> People.id
    -- Roles.movie_id -> Movies.id
    -- Roles.person_id -> People.id
    -- Genres.movie_id -> Movies.id
    """
    
    prompt = (
        f"Based on the following PostgreSQL database schema, write a SQL query to answer the user's question.\n"
        f"Remember to only return a SELECT query and nothing else. Do not use any comments.\n"
        f"Do not use INSERT, UPDATE, or DELETE statements.\n\n"
        f"For movie titles, director names, and actor names, use the LIKE operator for partial matches, and make sure the query is case insensitive.\n"
        f"Example: `... WHERE lower(title) LIKE '%avengers%'`\n\n"
        f"{schema}\n\n"
        f"User query: \"{user_query}\"\n"
        f"SQL query:"
    )
    
    try:
        sql_query = call_gemini_api(prompt)
        if not sql_query:
            return None
        
        # Clean up the output to ensure only the SQL query remains
        if sql_query.upper().startswith("SQL"):
            sql_query = sql_query[3:].strip()
        return sql_query
    except Exception as e:
        print(f"An error occurred with Gemini for NL-to-SQL: {e}")
        return None

def run_sql_query(sql_query):
    """
    Connects to the database and executes the provided SQL query.
    Returns a dict with the results or error message.
    """
    conn = None
    try:
        conn = connect_to_postgres()
        if not conn:
            return {'error': 'Could not connect to database.'}
            
        cur = conn.cursor()
        cur.execute(sql_query)
        
        # Check if the query returns data
        if cur.description is not None:
            column_names = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return {'sql_query': sql_query, 'columns': column_names, 'rows': rows}
        else:
            cur.close()
            conn.close()
            return {'message': 'Query executed successfully with no results to return.'}
            
    except Exception as e:
        print(f"Database error: {e}")
        return {'error': f"Database error: {e}"}

# --- Main Lambda Handler ---
def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    1. Parses the user query from the event.
    2. Attempts to answer using NL-to-SQL and the movie database.
    3. If no SQL result, falls back to RAG workflow.
    4. Returns the answer as a JSON response.
    """
    # Check for required API keys
    if not GEMINI_API_KEY:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": "Gemini API key is not configured."})
        }
    if not NOMIC_API_KEY:
        return {
            'statusCode': 500,
            'body': json.dumps({"error": "Nomic API key is not configured."})
        }
        
    try:
        # Parse the user query from the event (supports API Gateway and test events)
        if 'body' in event and isinstance(event['body'], str):
            body = json.loads(event.get('body', '{}'))
            user_query = body.get('query')
        else:
            user_query = event.get('query')

        if not user_query:
            return {
                'statusCode': 400,
                'body': json.dumps({"error": "Missing 'query' in request body"})
            }

        # 1. Try NL-to-SQL workflow first
        sql_query = convert_nl_to_sql(user_query)
        if sql_query and sql_query.upper().startswith("SELECT"):
            print(f"Generated SQL query: {sql_query}")
            sql_result = run_sql_query(sql_query)
            
            if 'error' not in sql_result and 'rows' in sql_result and sql_result['rows']:
                # SQL query was successful and returned data
                return {
                    'statusCode': 200,
                    'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                    'body': json.dumps(sql_result, default=str)
                }

        # 2. If SQL query fails or returns no data, fall back to RAG
        print("SQL query failed or returned no data. Falling back to RAG workflow.")
        
        # Load the FAISS index from S3 (cached for warm starts)
        faiss_index = load_faiss_index_from_s3()
        if faiss_index is None:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": "Failed to load FAISS index."})
            }
        conn = connect_to_postgres()
        if not conn:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": "Failed to connect to database for RAG."})
            }
            
        rag_answer, rag_context = perform_rag_query(user_query, faiss_index, conn)
        conn.close()
        
        if rag_answer:
            return {
                'statusCode': 200,
                'headers': {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
                'body': json.dumps({"rag_answer": rag_answer}, default=str)
            }
        else:
            return {
                'statusCode': 500,
                'body': json.dumps({"error": "Failed to generate a response via RAG."})
            }

    except Exception as e:
        print(f"An unexpected error occurred in the handler: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({"error": "An unexpected error occurred."})
        }