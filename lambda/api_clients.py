import json

import numpy as np
import requests

from config import GEMINI_API_KEY, NOMIC_API_KEY


def call_gemini_api(prompt):
    """
    Makes a direct HTTP request to the Gemini API for text generation.
    """
    if GEMINI_API_KEY is None:
        raise ValueError("GEMINI_API_KEY environment variable is not set.")

    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-3.1-flash-lite-preview:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    print("calling gemini")
    try:
        response = requests.post(api_url, headers={"Content-Type": "application/json"}, data=json.dumps(payload))
        response.raise_for_status()
        result = response.json()
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Error calling Gemini API: {e}")
        return None


def get_nomic_embedding(text, task_type="search_query"):
    """
    Generates an embedding for a given text using the Nomic API.
    """
    if NOMIC_API_KEY is None:
        raise ValueError("NOMIC_API_KEY environment variable is not set.")

    api_url = "https://api-atlas.nomic.ai/v1/embedding/text"
    headers = {
        "Authorization": f"Bearer {NOMIC_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "texts": [text],
        "model": "nomic-embed-text-v1.5",
        "task_type": task_type,
    }

    try:
        response = requests.post(api_url, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        embedding_data = response.json()
        return np.array(embedding_data["embeddings"][0]).astype("float32")
    except requests.exceptions.RequestException as e:
        print(f"Error calling Nomic API: {e}")
        return None
