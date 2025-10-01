
# Standard library imports
import re

# Third-party imports
import psycopg2
from imdb import Cinemagoer
from cinemagoerng import web


def clean_votes(votes_str):
    """
    Remove non-digit characters from a vote string and convert to integer.
    Args:
        votes_str (str|int): The vote count as a string or integer.
    Returns:
        int: Cleaned vote count as integer.
    """
    if isinstance(votes_str, str):
        return int(re.sub(r'[^\d]', '', votes_str))
    return votes_str


def setup_database(db_name, user, password, host, port):
    """
    Connect to PostgreSQL and create required tables if they do not exist.
    Returns:
        tuple: (connection, cursor)
    """
    conn = psycopg2.connect(
        dbname=db_name,
        user=user,
        password=password,
        host=host,
        port=port
    )
    cur = conn.cursor()

    # Create Movies table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS Movies (
            id TEXT PRIMARY KEY,
            title TEXT,
            year INTEGER,
            kind TEXT,
            rating REAL,
            votes INTEGER,
            runtime INTEGER,
            plot_summary TEXT
        )
    ''')

    # Create People table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS People (
            id TEXT PRIMARY KEY,
            name TEXT
        )
    ''')

    # Create linking tables
    cur.execute('''CREATE TABLE IF NOT EXISTS Directors (
        movie_id TEXT, person_id TEXT, PRIMARY KEY (movie_id, person_id)
    )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS Roles (
        movie_id TEXT, person_id TEXT, role TEXT, PRIMARY KEY (movie_id, person_id)
    )''')
    cur.execute('''CREATE TABLE IF NOT EXISTS Genres (
        movie_id TEXT, genre TEXT, PRIMARY KEY (movie_id, genre)
    )''')

    conn.commit()
    return conn, cur


def populate_database():
    """
    Fetch IMDb Top 250 movies and populate the PostgreSQL database.
    """
    ia = Cinemagoer()

    # Update these parameters as needed
    conn, cur = setup_database(
        db_name='movies_pg_250',
        user='postgres',
        password='<pass>',  # TODO: Replace with your actual password
        host='localhost',
        port=5432
    )

    print("Fetching Top 250 movies...")
    top_250 = ia.get_top250_movies()

    for i, movie_summary in enumerate(top_250[:250]):
        print(f"Processing movie {i+1}/250: {movie_summary['title']}...")
        movie_id = movie_summary.movieID

        # Fetch detailed movie info using cinemagoerng for full credits
        movie = web.get_title("tt" + movie_id)

        # --- Insert Directors into People and Directors tables ---
        if movie.directors:
            for director in movie.directors:
                cur.execute(
                    "INSERT INTO People (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (director.imdb_id, director.name)
                )
                cur.execute(
                    "INSERT INTO Directors (movie_id, person_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (movie_id, director.imdb_id)
                )

        # --- Insert Cast (top 5) into People and Roles tables ---
        if movie.cast:
            for actor in movie.cast[:5]:
                cur.execute(
                    "INSERT INTO People (id, name) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                    (actor.imdb_id, actor.name)
                )
                role = actor.characters[0] if hasattr(actor, 'characters') and actor.characters else 'N/A'
                cur.execute(
                    "INSERT INTO Roles (movie_id, person_id, role) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (movie_id, actor.imdb_id, role)
                )

        # --- Insert Genres ---
        if movie.genres:
            for genre in movie.genres:
                cur.execute(
                    "INSERT INTO Genres (movie_id, genre) VALUES (%s, %s) ON CONFLICT DO NOTHING",
                    (movie_id, genre)
                )

        # --- Insert/Update Movie record ---
        plot_summary = (
            movie.plot_summaries['en-US'][0]
            if movie.plot_summaries and 'en-US' in movie.plot_summaries else ''
        )
        runtime = int(movie.runtime) if movie.runtime else 0
        votes = clean_votes(movie.vote_count) if hasattr(movie, 'vote_count') else 0
        rating = float(movie.rating) if hasattr(movie, 'rating') else 0.0

        cur.execute('''
            INSERT INTO Movies (id, title, year, kind, rating, votes, runtime, plot_summary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                title=EXCLUDED.title,
                year=EXCLUDED.year,
                kind=EXCLUDED.kind,
                rating=EXCLUDED.rating,
                votes=EXCLUDED.votes,
                runtime=EXCLUDED.runtime,
                plot_summary=EXCLUDED.plot_summary
        ''', (
            movie_id,
            movie.title,
            movie.year,
            movie.type_id,
            rating,
            votes,
            runtime,
            plot_summary
        ))
        conn.commit()

    print("Database population complete.")
    conn.close()


if __name__ == '__main__':
    populate_database()


# ---
# Example SQL for table creation (for reference):
#
# CREATE TABLE Movies (
#     id TEXT PRIMARY KEY,
#     title TEXT,
#     year INTEGER,
#     kind TEXT,
#     rating REAL,
#     votes INTEGER,
#     runtime INTEGER,
#     plot_summary TEXT
# );
#
# CREATE TABLE People (
#     id TEXT PRIMARY KEY,
#     name TEXT
# );
#
# CREATE TABLE Directors (
#     movie_id TEXT,
#     person_id TEXT,
#     PRIMARY KEY (movie_id, person_id),
#     FOREIGN KEY (movie_id) REFERENCES Movies(id),
#     FOREIGN KEY (person_id) REFERENCES People(id)
# );
#
# CREATE TABLE Roles (
#     movie_id TEXT,
#     person_id TEXT,
#     role TEXT,
#     PRIMARY KEY (movie_id, person_id),
#     FOREIGN KEY (movie_id) REFERENCES Movies(id),
#     FOREIGN KEY (person_id) REFERENCES People(id)
# );
#
# CREATE TABLE Genres (
#     movie_id TEXT,
#     genre TEXT,
#     PRIMARY KEY (movie_id, genre),
#     FOREIGN KEY (movie_id) REFERENCES Movies(id)
# );
# ---