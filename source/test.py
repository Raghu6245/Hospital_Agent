from dotenv import load_dotenv
import os

load_dotenv("/Users/raghuvarun/Desktop/chatbot/source/.env")  # Explicit full path to .env
print(os.getenv("OPENAI_API_KEY"))  # Replace "API_KEY" with your actual variable name
