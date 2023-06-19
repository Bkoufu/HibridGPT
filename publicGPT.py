import openai
import os

def get_response_from_openai(query):
    openai.api_key = os.getenv("OPENAI_API_KEY") # set your API key
    response = openai.Completion.create(engine="text-davinci-002", prompt=query, max_tokens=150)
    return response.choices[0].text.strip()
