# from fastapi import FastAPI

# app = FastAPI()

# @app.get("/")
# def read_root():
#     return {"message": "Hello, FastAPI!"}


import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import shutil
import tempfile
import asyncio
import torch # Import torch
from transformers import pipeline
import librosa
from langid import classify
import soundfile as sf # Import soundfile for writing preprocessed audio
# from langdetect import detect
# from langchain.chat_models import ChatGoogleGemini
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()
api_key = os.getenv("API_KEY")

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Langchain components
# llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash")
llm = ChatGoogleGenerativeAI(
    google_api_key=api_key,
    model="gemini-2.5-flash"
)
prompt = PromptTemplate(
    input_variables=["country"],
    template="What is the capital of {country}? Only return the capital city."
)

# Refactor LLMChain to use RunnableSequence and StrOutputParser
llm_chain = prompt | llm | StrOutputParser()

# Initialize Whisper model for Speech-to-Text
# Using a smaller model for local execution to reduce download size and memory usage
# You might need to adjust this based on available resources and desired accuracy
print("Loading Whisper models for Speech-to-Text and Translation. This may take some time on first run...")
# Initialize ASR pipeline for transcription in original language
asr_pipe = pipeline(model="openai/whisper-tiny", device=0 if torch.cuda.is_available() else -1)
# Initialize Translation pipeline for direct translation to English
translation_pipe = pipeline(model="openai/whisper-tiny", device=0 if torch.cuda.is_available() else -1, generate_kwargs={"task": "translate", "language": "en"})

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/upload_audio")
async def upload_audio(request: Request, audio_file: UploadFile = File(...)):
    try:
        # Save the uploaded file temporarily
        # Use the original file extension to ensure compatibility with various audio formats
        original_filename = audio_file.filename
        file_extension = os.path.splitext(original_filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
            shutil.copyfileobj(audio_file.file, tmp_file)
            tmp_file_path = tmp_file.name

        # Load audio with librosa to ensure consistent sampling rate and format
        # Whisper models typically expect 16kHz mono audio
        print("Loading audio with librosa for preprocessing...")
        audio, sr = librosa.load(tmp_file_path, sr=16000, mono=True)
        
        # Save the preprocessed audio to a new temporary file for Whisper
        preprocessed_tmp_file_path = tmp_file_path.replace(file_extension, f"_preprocessed{file_extension}")
        sf.write(preprocessed_tmp_file_path, audio, sr)
        
        # Update tmp_file_path to point to the preprocessed file
        tmp_file_path = preprocessed_tmp_file_path

        # Speech-to-Text using Whisper for original transcript
        print("Performing speech-to-text using Whisper for original transcript...")
        # Use chunking for longer audio files to prevent memory issues and improve performance
        whisper_result_original = asr_pipe(tmp_file_path, chunk_length_s=30, stride_length_s=5)
        original_transcript = whisper_result_original["text"]
        
        if not original_transcript:
            return templates.TemplateResponse("index.html", {"request": request, "result": "Could not transcribe audio. Please try again with a clearer recording."})

        # Language Detection
        print("Detecting language using langid...")
        try:
            # langid.classify returns a tuple (language_code, confidence)
            detected_language_code, _ = classify(original_transcript)
        except Exception as e:
            print(f"Langid detection failed: {e}. Defaulting to English.")
            detected_language_code = "en" # Default to English if detection fails

        translated_text = original_transcript
        if detected_language_code != "en":
            print(f"Translating from {detected_language_code} to English using Whisper...")
            # Use Whisper's translation pipeline for direct translation
            whisper_result_translated = translation_pipe(tmp_file_path, chunk_length_s=30, stride_length_s=5)
            translated_text = whisper_result_translated["text"]
        else:
            print("Audio is already in English.")

        # Summarize discussed points using LLM
        summary_prompt = PromptTemplate(
            input_variables=["text"],
            template="Summarize the following text and list out all the discussed points:\n\n{text}"
        )
        summary_chain = summary_prompt | llm | StrOutputParser()
        print("Summarizing discussed points...")
        summary = summary_chain.invoke({"text": translated_text})

        os.remove(tmp_file_path) # Clean up the preprocessed temporary file
        if 'preprocessed_tmp_file_path' in locals() and preprocessed_tmp_file_path != original_filename:
            # Only remove the original temporary file if it's different from the preprocessed one
            # and not the original uploaded file (which is handled by FastAPI)
            original_tmp_file_path = tmp_file_path.replace(f"_preprocessed{file_extension}", file_extension)
            if os.path.exists(original_tmp_file_path):
                os.remove(original_tmp_file_path)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {
                "original_transcript": original_transcript,
                "detected_language": detected_language_code,
                "translated_text": translated_text,
                "summary": summary
            }
        })

    except Exception as e:
        print(f"Error in upload_audio: {e}")
        return templates.TemplateResponse("index.html", {"request": request, "result": f"An error occurred: {e}"})

# Remove the old /chat endpoint as it's no longer relevant for the new agent
# @app.post("/chat")
# async def chat_with_capital_agent(request: Request, country_name: str = Form(...)):
#     try:
#         print(f"Received country name: {country_name}")
#         capital = llm_chain.invoke({"country": country_name})
#         print(f"Capital: {capital}")
#         return {"response": f"The capital of {country_name} is {capital}."}
#     except Exception as e:
#         print(f"Error in chat_with_capital_agent: {e}")
#         return {"response": f"Error: {e}"}
