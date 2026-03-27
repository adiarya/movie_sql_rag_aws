import psycopg2

from config import DB_HOST, DB_NAME, DB_PASSWORD, DB_USER


def connect_to_postgres():
    """
    Establishes a connection to the PostgreSQL database.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=5432,
        )
        print("Successfully connected to PostgreSQL database.")
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None


def create_movie_documents(conn, movie_ids):
    """
    Retrieves movie information from the database for the given IDs more efficiently.
    """
    cursor = conn.cursor()
    movie_documents = []

    if not movie_ids:
        return movie_documents

    padded_movie_ids = [str(mid).zfill(7) for mid in movie_ids]
    id_list_sql = ", ".join([f"'{mid}'" for mid in padded_movie_ids])

    cursor.execute(
        f"SELECT id, title, year, kind, rating, votes, runtime, plot_summary FROM public.movies WHERE id IN ({id_list_sql})"
    )
    movie_details = {row[0]: row[1:] for row in cursor.fetchall()}

    cursor.execute(f"SELECT movie_id, genre FROM public.genres WHERE movie_id IN ({id_list_sql})")
    genres_map = {}
    for movie_id, genre in cursor.fetchall():
        if movie_id not in genres_map:
            genres_map[movie_id] = []
        genres_map[movie_id].append(genre)

    cursor.execute(
        f"""
        SELECT d.movie_id, p.name FROM public.people p
        JOIN public.directors d ON p.id = d.person_id
        WHERE d.movie_id IN ({id_list_sql})
    """
    )
    directors_map = {}
    for movie_id, name in cursor.fetchall():
        if movie_id not in directors_map:
            directors_map[movie_id] = []
        directors_map[movie_id].append(name)

    cursor.execute(
        f"""
        SELECT r.movie_id, p.name, r.role FROM public.people p
        JOIN public.roles r ON p.id = r.person_id
        WHERE r.movie_id IN ({id_list_sql})
    """
    )
    cast_map = {}
    for movie_id, name, role in cursor.fetchall():
        if movie_id not in cast_map:
            cast_map[movie_id] = []
        cast_map[movie_id].append(f"{name} as {role}")

    for movie_id in padded_movie_ids:
        if movie_id not in movie_details:
            continue

        title, year, kind, rating, votes, runtime, plot_summary = movie_details[movie_id]
        genres = ", ".join(genres_map.get(movie_id, []))
        directors = ", ".join(directors_map.get(movie_id, []))
        cast = ", ".join(cast_map.get(movie_id, []))

        doc_text = (
            f"Title: {title}\\n"
            f"Year: {year}\\n"
            f"Type: {kind}\\n"
            f"Rating: {rating}/10 from {votes} votes.\\n"
            f"Runtime: {runtime} minutes\\n"
            f"Genre(s): {genres}\\n"
            f"Directed by: {directors}\\n"
            f"Cast: {cast}\\n"
            f"Plot Summary: {plot_summary}"
        )
        movie_documents.append({"id": movie_id, "content": doc_text})

    cursor.close()
    return movie_documents


def run_sql_query(sql_query):
    """
    Connects to the database and executes the provided SQL query.
    """
    conn = None
    try:
        conn = connect_to_postgres()
        if not conn:
            return {"error": "Could not connect to database."}

        cur = conn.cursor()
        cur.execute(sql_query)

        if cur.description is not None:
            column_names = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            cur.close()
            conn.close()
            return {"sql_query": sql_query, "columns": column_names, "rows": rows}

        cur.close()
        conn.close()
        return {"message": "Query executed successfully with no results to return."}

    except Exception as e:
        print(f"Database error: {e}")
        return {"error": f"Database error: {e}"}
