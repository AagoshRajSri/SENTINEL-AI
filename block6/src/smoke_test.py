import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


def verify_setup():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY not set in environment. "
            "Please check your .env file."
        )

    client = genai.Client(api_key=api_key)

    print("Sending request to Gemini API (temperature=0.0)...")

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents="Respond exactly with this text: 'Sentinel-AI setup verified.'",
        config=types.GenerateContentConfig(
            temperature=0.0,
        ),
    )

    print(f"Response: {response.text.strip()}")


if __name__ == "__main__":
    verify_setup()