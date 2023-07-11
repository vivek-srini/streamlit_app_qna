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

def transcript_audio(audio_bytes):
  r = sr.Recognizer()
# Reading Audio file as source
#  listening  the  аudiо  file  аnd  stоre  in  аudiо_text  vаriаble
  with sr.AudioFile(audio_bytes) as source:
    audio_text = r.listen(source)
# recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
   
        # using google speech recognition
    text = r.recognize_google(audio_text)
  return text   

# def read_from_db(db_name,table_name):
#     conn = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='plagueofdeath',
#     db=db_name
    
# )
#     cursor = conn.cursor()
#     query = f"SELECT * FROM {table_name}"
#     cursor.execute(query)
#     rows = cursor.fetchall()
#     column_names = [column[0] for column in cursor.description]
#     df = pd.DataFrame(rows, columns=column_names)
#     return df

# def write_df_to_db(df,db_name,table_name):
#     conn = mysql.connector.connect(
#     host='localhost',
#     user='root',
#     password='plagueofdeath',
#     db=db_name
    
# )
#     engine = create_engine(f'mysql+mysqlconnector://root:plagueofdeath@localhost/{db_name}')
   
#     df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
#     conn.close()
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
    
def langchain_response_without_prompt(db,question):
   
    prompt_template = """Answer in as much detail as possible but dont make things up. Only use the information in the context."""
    prompt_template = prompt_template + "\n" + """context: {context}
            question: {question}
            Helpful Answer: """
            
    prompt = PromptTemplate(template=prompt_template, input_variables=['context',"question"])
    type_kwargs = {"prompt": prompt}
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":int(3)})
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
      custom_params = st.checkbox('Use Custom Prompt and K')
      if custom_params:
        selected_language = st.selectbox('Select Language/மொழியை தேர்ந்தெடுங்கள்/भाषा चुने', languages)
        prompt_template = st.text_input("Please enter the prompt you would like to use")
        k = st.text_input("Please enter a value for k")
        require_audio = st.checkbox('I would rather ask a question orally')
        if require_audio:
            audio_bytes = audio_recorder()
            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
            user_question = transcript_audio(audio_bytes)
        else:
            user_question = st.text_input("Ask a question about your PDF:")
        if user_question and k:
          
          if selected_language == 'Tamil':
            user_question = translate_tamil_to_english(user_question)
          elif selected_language == 'Hindi':
            user_question = translate_hindi_to_english(user_question)
        
         
        # docs = knowledge_base.similarity_search(user_question)
        
        # llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
        # chain = load_qa_chain(llm, chain_type="stuff",)
        # with get_openai_callback() as cb:
        #   response = chain.run(input_documents=docs, question=user_question)
        #   print(cb)
        

       
          response = langchain_response(db, user_question, prompt_template, k)
          if selected_language=="Hindi":
            t5 = time.time()
            response = translate_english_to_hindi(response)
            st.write("Time taken for translation: ",time.time()-t5)
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
          st.write("Time taken for voiceover: ", t2-t1)
          st.write(response)
         
      else:
        selected_language = st.selectbox('Select Language/மொழியை தேர்ந்தெடுங்கள்/भाषा चुने', languages)
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
          orig_user_question = user_question 
          if selected_language == 'Tamil':
            user_question = translate_tamil_to_english(user_question)
          elif selected_language == 'Hindi':
            user_question = translate_hindi_to_english(user_question)
          t10 = time.time()
          response = langchain_response_without_prompt(db, user_question)
          st.write("Time taken by model: ",time.time()-t10)
          if selected_language=="Hindi":
            t5 = time.time()
            response = translate_english_to_hindi(response)
            st.write("Time taken for translation: ",time.time()-t5)
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
          st.write("Time taken for voiceover: ", t2-t1)
          st.write(response)
          # df = read_from_db("qna_streamlit","questions_answers")
          # df = df.append({"Question":orig_user_question,"Answer":response},ignore_index=True)
          # write_df_to_db(df,"qna_streamlit","questions_answers")

if __name__ == '__main__':
    main()

