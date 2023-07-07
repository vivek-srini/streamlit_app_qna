from dotenv import load_dotenv
import streamlit as st
import openai
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import os
openai_api_key = os.environ.get('OPENAI_API_KEY')
from mtranslate import translate

def translate_tamil_to_english(text):
    translated_text = translate(text, 'en', 'ta')
    return translated_text

def translate_hindi_to_english(text):
    translated_text = translate(text,'en','hi')
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
      embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
      # show user input
      languages = ['English', 'Tamil','Hindi']
      selected_language = st.selectbox('Select Language/மொழியை தேர்ந்தெடுங்கள்/भाषा चुने', languages)
      
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        if selected_language == 'Tamil':
            user_question = translate_tamil_to_english(user_question)
        elif selected_language == 'Hindi':
            user_question = translate_hindi_to_english(user_question)
        
         
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI(temperature=0,openai_api_key=openai_api_key)
        chain = load_qa_chain(llm, chain_type="stuff",)
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

if __name__ == '__main__':
    main()

