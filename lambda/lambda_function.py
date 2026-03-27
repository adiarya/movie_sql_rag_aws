import json
from config import DEFAULT_HEADERS, GEMINI_API_KEY, NOMIC_API_KEY
from db import connect_to_postgres, run_sql_query
from faiss_utils import load_faiss_index_from_s3
from nl2sql import convert_nl_to_sql
from rag import perform_rag_query

# --- Main Lambda Handler ---
def lambda_handler(event, context):
    """
    The main handler for the Lambda function. It parses the user query and
    routes it to either the RAG or NL-to-SQL workflow.
    """
    if not GEMINI_API_KEY:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Gemini API key is not configured."}),
        }

    if not NOMIC_API_KEY:
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Nomic API key is not configured."}),
        }

    try:
        # Handle both API Gateway proxy body and direct test event payloads.
        if "body" in event and isinstance(event["body"], str):
            body = json.loads(event.get("body", "{}"))
            user_query = body.get("query")
        else:
            user_query = event.get("query")

        if not user_query:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'query' in request body"}),
            }

        sql_query = convert_nl_to_sql(user_query)
        if sql_query and sql_query.upper().startswith("SELECT"):
            print(f"Generated SQL query: {sql_query}")
            sql_result = run_sql_query(sql_query)

            if "error" not in sql_result and "rows" in sql_result and sql_result["rows"]:
                return {
                    "statusCode": 200,
                    "headers": DEFAULT_HEADERS,
                    "body": json.dumps(sql_result, default=str),
                }

        print("SQL query failed or returned no data. Falling back to RAG workflow.")

        faiss_index = load_faiss_index_from_s3()
        print("\ndownloaded\n")
        if faiss_index is None:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Failed to load FAISS index."}),
            }

        conn = connect_to_postgres()
        if not conn:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Failed to connect to database for RAG."}),
            }

        rag_answer, rag_context = perform_rag_query(user_query, faiss_index, conn)
        conn.close()

        if rag_answer:
            return {
                "statusCode": 200,
                "headers": DEFAULT_HEADERS,
                "body": json.dumps({"rag_answer": rag_answer}, default=str),
            }

        return {
            "statusCode": 500,
            "body": json.dumps({"error": "Failed to generate a response via RAG."}),
        }

    except Exception as e:
        print(f"An unexpected error occurred in the handler: {e}")
        return {
            "statusCode": 500,
            "body": json.dumps({"error": "An unexpected error occurred."}),
        }