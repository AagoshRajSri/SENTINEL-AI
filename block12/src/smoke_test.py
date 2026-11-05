import os
from dotenv import load_dotenv
from groq import Groq


def verify_setup():
    load_dotenv()

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "CRITICAL: GROQ_API_KEY environment variable is missing. Please check your .env file."
        )

    client = Groq(api_key=groq_api_key)

    print("Sending request to Groq API...")
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "user", "content": "Respond exactly with this text: 'Sentinel-AI setup verified.'"}
        ],
        temperature=0.0
    )
    print(f"Response: {response.choices[0].message.content.strip()}")


if __name__ == "__main__":
    verify_setup()