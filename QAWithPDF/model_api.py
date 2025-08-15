import os
from dotenv import load_dotenv
import sys
import google.generativeai as genai
from llama_index.llms.gemini import Gemini
from QAWithPDF.exception import customexception
from logger import logging

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables or .env file.")

# Configure SDK
genai.configure(api_key=GOOGLE_API_KEY)

def load_model():
    """
    Loads the Gemini 2.0 Flash model using Google Generative AI API.
    """
    try:
        logging.info("Loading Gemini 2.0 Flash model...")
        model = Gemini(model="gemini-2.0-flash", temperature=0.2)
        logging.info("Gemini 2.0 Flash model loaded successfully.")
        return model
    except Exception as e:
        raise customexception(e, sys)

