from dotenv import load_dotenv
import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
import os
from langchain.chat_models import ChatOpenAI
openai_api_key = os.environ.get('OPENAI_API_KEY')
from mtranslate import translate
from gtts import gTTS
from io import BytesIO
import time
import pandas as pd
import mysql.connector
from mysql.connector import Error
from sqlalchemy import create_engine
import pandas as pd
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import speech_recognition as sr
from audiorecorder import audiorecorder
from pydub import AudioSegment
import subprocess
def transcript_english_audio(audio_bytes,selected_language):
  if os.path.exists('file.wav'):
    os.remove('file.wav')
  subprocess.call(['ffmpeg', '-i', audio_bytes,
                 'file.wav'])
  
  r = sr.Recognizer()
# Reading Audio file as source
#  listening  the  аudiо  file  аnd  stоre  in  аudiо_text  vаriаble
  with sr.AudioFile("file.wav") as source:
    audio_text = r.record(source)
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
   
        # using google speech recognition
    text = r.recognize_google(audio_text,language=selected_language[:2].lower()+"-IN")
  print(text)
  return text 

def create_audio_file(text,language):
  sound_file = BytesIO()
  tts = gTTS(text, lang=language)
  tts.write_to_fp(sound_file)
  return sound_file

def translate_tamil_to_english(text):
    translated_text = translate(text, 'en', 'ta')
    return translated_text
def langchain_response(db,question,prompt_template,k):
    
    prompt_template = prompt_template + "\n" + """context: {context}
            question: {question}
            Helpful Answer: """
            
    prompt = PromptTemplate(template=prompt_template, input_variables=['context',"question"])
    type_kwargs = {"prompt": prompt}
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":int(k)})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0,max_tokens=600,openai_api_key = openai_api_key), chain_type="stuff",retriever=retriever, chain_type_kwargs=type_kwargs)
    result = qa({"query": question})
    return result["result"]
    
def langchain_response_without_prompt(db,question,k):
   
    prompt_template = """Answer in as much detail as possible but dont make things up. Only use the information in the context."""
    prompt_template = prompt_template + "\n" + """context: {context}
            question: {question}
            Helpful Answer: """
            
    prompt = PromptTemplate(template=prompt_template, input_variables=['context',"question"])
    type_kwargs = {"prompt": prompt}
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":int(k)})
    qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(temperature=0,max_tokens=600,openai_api_key = openai_api_key), chain_type="stuff",retriever=retriever, chain_type_kwargs=type_kwargs)
    result = qa({"query": question})
    return result["result"]


def translate_hindi_to_english(text):
    translated_text = translate(text,'en','hi')
    return translated_text

def translate_english_to_hindi(text):
    translated_text = translate(text,'hi','en')
    return translated_text

def translate_english_to_tamil(text):
    translated_text = translate(text,'ta','en')
    return translated_text
def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = RecursiveCharacterTextSplitter(
        
        chunk_size=2000,
        chunk_overlap=0
        
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      t12 = time.time()
      embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
      
      db = FAISS.from_texts(chunks, embeddings)
    
      print("Time taken by Embeddings model: ",t12 - time.time())
      # show user input
      languages = ['English', 'Tamil','Hindi']
      custom_params = st.checkbox('Use Custom Prompt')
      if custom_params:
        selected_language = st.selectbox('Select Language/மொழியை தேர்ந்தெடுங்கள்/भाषा चुने', languages)
        prompt_template = st.text_input("Please enter the prompt you would like to use/தயவு செய்து நீங்கள் எங்களிடம் கேட்கும் prompt உள்ளிடவும்/कृपया वह prompt शामिल करें जो आप हमसे चाहेंगे")
        k = st.text_input("Please enter a value for k")
        require_audio = st.checkbox('I would rather ask a question orally/நான் வாய்மொழியாக ஒரு கேள்வியைக் கேட்பேன்/मैं मौखिक रूप से एक प्रश्न पूछना पसंद करूंगा')
        if require_audio:
            audio = audiorecorder("Click to record/பதிவு செய்ய கிளிக் செய்யவும்/रिकॉर्ड करने के लिए क्लिक करें", "Click to stop once done recording.../பதிவு செய்தவுடன் நிறுத்த கிளிக் செய்யவும்.../रिकॉर्डिंग पूरी हो जाने पर रोकने के लिए क्लिक करें...")
            if len(audio)>0:
                st.audio(audio.tobytes())
                wav_file = open("audio.mp3", "wb")
                wav_file.write(audio.tobytes())
        if require_audio and len(audio)>0:
          
          user_question = transcript_english_audio("audio.mp3",selected_language)
          string_to_write = "Your Question/உங்கள் கேள்வி/आपका प्रश्न: "
          st.write(string_to_write)
          st.write(user_question)
          
        else:
          user_question = st.text_input("Ask a question about your PDF:/உங்கள் PDF பற்றிய கேள்வியைக் கேளுங்கள்:/अपने PDF के बारे में प्रश्न पूछें:")
        if user_question:
          
          if selected_language == 'Tamil':
            user_question = translate_tamil_to_english(user_question)
            st.write(user_question)
          elif selected_language == 'Hindi':
            user_question = translate_hindi_to_english(user_question)
            st.write(user_question)
        
         
        # docs = knowledge_base.similarity_search(user_question)
        
        # llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
        # chain = load_qa_chain(llm, chain_type="stuff",)
        # with get_openai_callback() as cb:
        #   response = chain.run(input_documents=docs, question=user_question)
        #   print(cb)
        

       
          response = langchain_response(db, user_question, prompt_template,k)
          if selected_language=="Hindi":
            t5 = time.time()
            response = translate_english_to_hindi(response)
           
            audio_file = create_audio_file(response,"hi")
          elif selected_language=="Tamil":
            response = translate_english_to_tamil(response)
            audio_file = create_audio_file(response,"ta")
          else:
            response = response
            audio_file = create_audio_file(response,"en")
          t1 = time.time()
          st.audio(audio_file)
          t2 = time.time()
         
          response_string = "Answer/பதில்/उत्तर: "
          st.write(response_string)
          st.write(response)
         
      else:
        selected_language = st.selectbox('Select Language/மொழியை தேர்ந்தெடுங்கள்/भाषा चुने', languages)
        require_audio = st.checkbox('I would rather ask a question orally/நான் வாய்மொழியாக ஒரு கேள்வியைக் கேட்பேன்/मैं मौखिक रूप से एक प्रश्न पूछना पसंद करूंगा')
        k = st.text_input("Enter a value for k")
        if require_audio:
            audio = audiorecorder("Click to record/பதிவு செய்ய கிளிக் செய்யவும்/रिकॉर्ड करने के लिए क्लिक करें", "Click to stop once done recording.../பதிவு செய்தவுடன் நிறுத்த கிளிக் செய்யவும்.../रिकॉर्डिंग पूरी हो जाने पर रोकने के लिए क्लिक करें...")
            if len(audio)>0:
                # st.audio(audio.tobytes())
                wav_file = open("audio.mp3", "wb")
                wav_file.write(audio.tobytes())
        if require_audio and len(audio)>0:
          
          user_question = transcript_english_audio("audio.mp3",selected_language)
          string_to_write = "Your Question/உங்கள் கேள்வி/आपका प्रश्न: "
          st.write(string_to_write)
          st.write(user_question)
        else:
          user_question = st.text_input("Ask a question about your PDF:/உங்கள் PDF பற்றிய கேள்வியைக் கேளுங்கள்:/अपने PDF के बारे में प्रश्न पूछें:")
        if user_question:
          orig_user_question = user_question 
          if selected_language == 'Tamil':
            user_question = translate_tamil_to_english(user_question)
          elif selected_language == 'Hindi':
            user_question = translate_hindi_to_english(user_question)
          t10 = time.time()
          response = langchain_response_without_prompt(db, user_question,k)
         
          if selected_language=="Hindi":
            t5 = time.time()
            response = translate_english_to_hindi(response)
            
            audio_file = create_audio_file(response,"hi")
          elif selected_language=="Tamil":
            response = translate_english_to_tamil(response)
            audio_file = create_audio_file(response,"ta")
          else:
            response = response
            audio_file = create_audio_file(response,"en")
          t1 = time.time()
          st.audio(audio_file)
          t2 = time.time()
         
          response_string = "Answer/பதில்/उत्तर: "
          st.write(response_string)
          st.write(response)
          # df = read_from_db("qna_streamlit","questions_answers")
          # df = df.append({"Question":orig_user_question,"Answer":response},ignore_index=True)
          # write_df_to_db(df,"qna_streamlit","questions_answers")

if __name__ == '__main__':
    main()

