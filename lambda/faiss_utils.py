import boto3
import faiss

from config import FAISS_KEY, FAISS_PATH, S3_BUCKET

s3_client = boto3.client("s3")
cached_faiss_index = None


def load_faiss_index_from_s3():
    """
    Downloads the FAISS index from S3 to the Lambda /tmp directory.
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
