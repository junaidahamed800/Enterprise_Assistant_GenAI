import os
import streamlit as st
import google.generativeai as genai

def get_api_key():
    try:
        return st.secrets["GOOGLE_API_KEY"]  # Streamlit Cloud
    except:
        return os.getenv("GOOGLE_API_KEY")   # Local .env

genai.configure(api_key=get_api_key())

model = genai.GenerativeModel("gemini-2.5-flash")

def generate_answer(prompt):
    response = model.generate_content(prompt)
    return response.text