# PostgreSQL IMDb Movie Database Builder

This project provides a Python script to create and populate a PostgreSQL database with IMDb Top 250 movie data, including directors, cast, genres, and plot summaries.

## Features
- Connects to a PostgreSQL database and creates tables for movies, people, directors, roles, and genres
- Fetches IMDb Top 250 movies using Cinemagoer and CinemagoerNG
- Populates the database with detailed movie, director, cast, and genre information
- Cleans and normalizes vote counts and other fields

## Requirements
- Python 3.7+
- PostgreSQL server
- Python packages:
  - `psycopg2`
  - `imdbpy` (Cinemagoer)
  - `cinemagoerng`

Install dependencies:
```
pip install psycopg2 imdbpy cinemagoerng
```

## Usage
1. Update the database connection parameters in `populate_database()` (database name, user, password, host, port).
2. Run the script:
```
python postgres_movie_db.py
```
3. The script will create the necessary tables (if they do not exist) and populate them with IMDb Top 250 data.

## Database Schema
- **Movies**: id, title, year, kind, rating, votes, runtime, plot_summary
- **People**: id, name
- **Directors**: movie_id, person_id
- **Roles**: movie_id, person_id, role
- **Genres**: movie_id, genre

## Example Table Creation SQL
See the bottom of `postgres_movie_db.py` for example SQL DDL statements.

## Notes
- The script uses Cinemagoer for the id of top 250 movies, and CinemagoerNG for detailed movie credits.
- Only the top 5 cast members are stored for each movie (for brevity).
- Update the database password and connection info before running.

## License
MIT License
