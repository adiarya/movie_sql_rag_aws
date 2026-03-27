import os

# Read from environment variables for security.
DB_NAME = os.environ.get("DB_NAME")
DB_USER = os.environ.get("DB_USER")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
DB_HOST = os.environ.get("DB_HOST")
S3_BUCKET = os.environ.get("S3_BUCKET")
FAISS_KEY = os.environ.get("FAISS_KEY")
FAISS_PATH = "/tmp/faiss_index3.bin"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
NOMIC_API_KEY = os.environ.get("NOMIC_API_KEY")

DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "http://hosted-movie-qa.s3-website-us-east-1.amazonaws.com",
}
