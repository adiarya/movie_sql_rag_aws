# Movie SQL + RAG AWS Lambda

This AWS Lambda function provides a hybrid natural language question answering system for a movie database using both:
- **NL-to-SQL**: Converts natural language queries to SQL and queries a PostgreSQL database.
- **RAG (Retrieval-Augmented Generation)**: Uses vector search (FAISS) and a large language model (Gemini) to answer questions when SQL is insufficient.

---

## Features

- **Natural Language to SQL**: Converts user queries to SQL using Gemini LLM and fetches results from PostgreSQL.
- **RAG Workflow**: If SQL fails or returns no data, retrieves relevant documents using FAISS vector search and generates an answer using Gemini LLM.
- **Embeddings**: Uses Nomic API for text embeddings.
- **Efficient Database Access**: Batch queries for movie details, genres, directors, and cast.
- **S3 Integration**: Loads FAISS index from S3 for scalable vector search.
- **API Ready**: Designed to be triggered by API Gateway for RESTful access.

---

## Environment Variables

Set the following environment variables for the Lambda function:

- `DB_NAME` – PostgreSQL database name
- `DB_USER` – PostgreSQL username
- `DB_PASSWORD` – PostgreSQL password
- `DB_HOST` – PostgreSQL host
- `S3_BUCKET` – S3 bucket containing the FAISS index
- `FAISS_KEY` – S3 key (path) to the FAISS index file
- `GEMINI_API_KEY` – API key for Gemini LLM
- `NOMIC_API_KEY` – API key for Nomic embedding service

---

## Dependencies

- `boto3`
- `faiss`
- `numpy`
- `requests`
- `psycopg2`
- `json`
- `os`

---

## How It Works

1. **Receives a query** via API Gateway (as a JSON payload with a `query` field).
2. **Attempts NL-to-SQL**:
   - Converts the query to SQL using Gemini LLM.
   - Runs the SQL on the movie database.
   - If results are found, returns them.
3. **If SQL fails or returns no data**:
   - Embeds the query using Nomic API.
   - Searches the FAISS vector index for similar movies.
   - Retrieves movie details from the database.
   - Constructs a context and prompts Gemini LLM for a natural language answer.
   - Returns the generated answer.
4. **Handles errors** gracefully and returns informative messages.

---

## Example Event

```json
{
  "body": "{\"query\": \"Who directed The Matrix?\"}"
}
```

---

## API Response

- **SQL Success:**
  ```json
  {
    "sql_query": "SELECT T2.name FROM Movies AS T1 INNER JOIN Directors AS T3 ON T1.id = T3.movie_id INNER JOIN People AS T2 ON T3.person_id = T2.id WHERE T1.title ILIKE '%The Matrix%'",
    "columns": ["director"],
    "rows": [["Lana Wachowski"], ["Lilly Wachowski"]]
  }
  ```
## Example Event

```json
{
  "body": "{\"query\": \"What is that movie about a killer robot sent from the future?\"}"
}
```

---

## API Response

- **RAG Success:**
  ```json
  {
    "rag_answer": "That sounds like **The Terminator** (1984)..."
  }
  ```

---

## Deployment

1. Package all dependencies (including FAISS and psycopg2) with your Lambda deployment.
2. Set the required environment variables.
3. Deploy the Lambda function and connect it to API Gateway for RESTful access.

---

## Notes

- Ensure your Lambda has network access to the PostgreSQL database and S3 bucket (VPC, security groups, IAM roles).
- The `/tmp` directory is used for temporary storage of the FAISS index.
- For production, restrict CORS headers to your frontend domain.

---

## License

MIT License (or your project’s license)
