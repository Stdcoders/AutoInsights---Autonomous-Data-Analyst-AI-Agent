# Test that both models work
import google.generativeai as genai
import os 
from dotenv import load_dotenv
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Test Pro model
pro_model = genai.GenerativeModel('gemini-1.5-pro')
response = pro_model.generate_content("Hello, are you working?")
print("Pro model:", response.text)

# Test Flash model  
flash_model = genai.GenerativeModel('gemini-1.5-flash')
response = flash_model.generate_content("Hello, are you working?")
print("Flash model:", response.text)