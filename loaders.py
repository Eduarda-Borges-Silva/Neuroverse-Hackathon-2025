import os
from time import sleep
import streamlit as st
from langchain_community.document_loaders import (WebBaseLoader, 
                                                  YoutubeLoader, 
                                                  CSVLoader, 
                                                  PyPDFLoader, 
                                                  TextLoader)

from fake_useragent import UserAgent

def carrega_sites(url):
    documentos = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader_site = WebBaseLoader(url, raise_for_status = True)
            lista_documentos = loader_site.load()
            documentos = '\n\n'.join([doc.page_content for doc in lista_documentos])
            break
        except:
            print(f'Erro ao Carregar o site {i + 1}')
            sleep(3)
    if documentos == '':
        st.error('Não foi Possivel carregar o site')
        st.stop()
    return documentos

def carrega_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info = False, language = ['pt'])
    lista_documentos = loader.load()
    documentos = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documentos


def carrega_csv(caminho):
    loader = CSVLoader(caminho)
    lista_documentos = loader.load()
    documentos = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documentos

def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documentos = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documentos

def carrega_txt(caminho):
    loader = TextLoader(caminho)
    lista_documentos = loader.load()
    documentos = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documentos