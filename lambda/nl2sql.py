from api_clients import call_gemini_api
from config import GEMINI_API_KEY


def convert_nl_to_sql(user_query):
    """
    Sends the user's natural language query to the Gemini API and
    returns the generated SQL query.
    """
    if not GEMINI_API_KEY:
        return None

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

    prompt = f"""
    Based on the following PostgreSQL database schema, write a well-formatted SQL query to answer the user's question.
    Remember to only return a SELECT query and nothing else. Do not use any comments.
    Do not use INSERT, UPDATE, or DELETE statements.

    For movie titles, director names, and actor names, use the ILIKE operator for partial matches, and make sure the query is case insensitive.
    Example: `... WHERE title ILIKE '%avengers%'`

    {schema}

    User query: "{user_query}"
    SQL query:
    """

    try:
        sql_query = call_gemini_api(prompt)
        if not sql_query:
            return None

        if sql_query.upper().startswith("SQL"):
            sql_query = sql_query[3:].strip()
        return sql_query
    except Exception as e:
        print(f"An error occurred with Gemini for NL-to-SQL: {e}")
        return None
