import os
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
import src.constants as constants
import psycopg2
import uuid
import numpy as np
import heapq
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def get_chunks_embeddings_from_docs(docs_dir_path: str, openai_api_key: str):
    # Load documents
    docs = load_docs(docs_dir_path)
    # Clean documents
    docs = clean_docs(docs)
    # Split documents into chunks
    docs_chunks = split_docs(docs)
    # Embed chunks
    docs_chunks_embeddings = embed_docs_chunks(docs_chunks, openai_api_key)
    # Name of the processed documents
    docs_names = list(docs_chunks.keys())
    # Return the chunks and embeddings
    return docs_names, docs_chunks, docs_chunks_embeddings

def load_docs(docs_dir_path: str):
    # Dictionary of documents to be loaded. Key is the document name, value is the document.
    # Each document is a list of pages.
    docs = {}
    # List of document names
    file_names = sorted(os.listdir(docs_dir_path))
    # Load documents
    for file_name in file_names:
        # Notify user about the document being loaded
        print(f"\nLoading document: {file_name}")
        # Load document
        doc_loader = PyPDFLoader(os.path.join(docs_dir_path, file_name))
        docs[file_name] = doc_loader.load()
    # Notify user about the document being loaded
    print(f"\nLoaded {len(docs)} documents.\n")
    # Return the documents
    return docs

def clean_docs(docs: dict[str, list[Document]]):
    # Remove newlines, non-breaking spaces, and hyphens from the document
    for doc_name, doc in docs.items():
        # Notify user about the document being cleaned
        print(f"\nCleaning document: {doc_name}")
        # Clean the document
        for page in doc:
            page.page_content = page.page_content.replace('\n', ' ').replace('\xa0', ' ').replace('\xad', '')
    # Notify user about the document being cleaned
    print(f"\nCleaned {len(docs)} documents.\n")
    # Return the cleaned documents
    return docs

def split_docs(docs: dict[str, list[Document]]):
    # Dictionary of split documents. Key is the document name, value is the split document.
    # Each split document is a list of chunks.
    docs_chunks = {}
    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=constants.TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=constants.TEXT_SPLITTER_CHUNK_OVERLAP
    )
    # Split documents into chunks
    for doc_name, doc in docs.items():
        # Notify user about the document being split
        print(f"\nSplitting document: {doc_name}")
        # Split the document
        docs_chunks[doc_name] = text_splitter.split_documents(doc)
    # Notify user about the document being split
    print(f"\nSplit {len(docs)} documents.\n")
    # Return the split documents
    return docs_chunks

def embed_docs_chunks(docs_chunks: dict[str, list[Document]], openai_api_key: str):
    # Dictionary of split document embeddings. Key is the document name, value is the split document embedding.
    # Each split document embedding is a list of embedding vectors.
    docs_chunks_embeddings = {}
    # Embed chunks
    for doc_name, doc_chunks in docs_chunks.items():
        # Notify user about the document being embedded
        print(f"\nEmbedding document: {doc_name}")
        # Embed the document
        docs_chunks_embeddings[doc_name] = [get_text_embedding(chunk.page_content, constants.OPENAI_EMBEDDING_MODEL, openai_api_key) for chunk in doc_chunks]
    # Notify user about the document being embedded
    print(f"\nEmbedded {len(docs_chunks)} documents.\n")
    # Return the embedded documents
    return docs_chunks_embeddings

def get_text_embedding(text: str, model: str, openai_api_key: str):
   # Instantiate OpenAI client
   client = OpenAI(api_key=openai_api_key)
   # Create and return embedding of given text
   return client.embeddings.create(input=[text], model=model).data[0].embedding

def setup_db():
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=constants.POSTGRES_HOST,
            port=constants.POSTGRES_PORT,
            user=constants.POSTGRES_USER,
            password=constants.POSTGRES_PASSWORD
        )
    except psycopg2.Error as e:
        raise e

    # Create a cursor
    cur = conn.cursor()

    # Isolation level autocommit
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)

    # Create the DB if not exists yet
    cur.execute(f"CREATE DATABASE IF NOT EXISTS {constants.POSTGRES_DATABASE}")

    # Close connection to the DB
    cur.close(), conn.close()

def connect_to_db():

    try:
        # Connect to the database
        conn = psycopg2.connect(
            host=constants.POSTGRES_HOST,
            port=constants.POSTGRES_PORT,
            user=constants.POSTGRES_USER,
            password=constants.POSTGRES_PASSWORD,
            database=constants.POSTGRES_DATABASE
        )
    except psycopg2.Error as e:
        raise e

    # Create a cursor
    cur = conn.cursor()

    # Return the connection and cursor
    return conn, cur

def insert_docs_into_db(
    docs_names: list[str],
    docs_chunks: dict[str, list[Document]],
    docs_embeddings: dict[str, list[list[float]]],
    conn: psycopg2.extensions.connection,
    cur: psycopg2.extensions.cursor):

    # Insert processed documents into the DB
    for doc_name in docs_names:

        # Get the chunks and embeddings for the current document
        chunks = docs_chunks[doc_name]
        embeddings = docs_embeddings[doc_name]

        # Insert the chunks and embeddings into the DB
        for chunk, embedding in zip(chunks, embeddings):

            # Check if the embedding already exists in the DB
            query = "SELECT * FROM embedding WHERE chunk = %s"
            cur.execute(query, (chunk.page_content,))
            query_result = cur.fetchall()
            if len(query_result) > 0:
                continue

            # Insert the new tuple
            id = uuid.uuid4()
            cur.execute(constants.DB_INSERT_TUPLE_SQL_COMMAND, (id, embedding, chunk.page_content))

    # Commit changes to the DB
    conn.commit()

def split_text(text: str):
    # Text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=constants.TEXT_SPLITTER_CHUNK_SIZE,
        chunk_overlap=constants.TEXT_SPLITTER_CHUNK_OVERLAP
    )
    return text_splitter.split_text(text)

def get_embeddings_chunks_from_db(cur: psycopg2.extensions.cursor):
    # Get all tuples from DB
    cur.execute("SELECT embedding, chunk FROM embedding")
    db_tuples = cur.fetchall()
    # Extract embeddings and chunks
    db_embeddings = [tuple[0] for tuple in db_tuples]
    db_chunks = [tuple[1] for tuple in db_tuples]
    # Return the embeddings and chunks
    return db_embeddings, db_chunks

def get_most_relevant_chunks(question_embeddings: list[list[float]], db_embeddings: list[list[float]], db_chunks: list[str]):
    # Compute cosine similarity between question embeddings and DB embeddings
    cosine_similarity_scores = [cosine_similarity(question_embeddings, db_embedding) for db_embedding in db_embeddings]
    # Get the indices of the most relevant chunks
    if constants.SEMANTIC_SEARCH_TOP_K > len(cosine_similarity_scores) or constants.SEMANTIC_SEARCH_TOP_K <= 0:
        raise ValueError("k must be greater than 0 and less than or equal to the length of the list")
    most_relevant_chunks_indices = [cosine_similarity_scores.index(i) for i in heapq.nlargest(constants.SEMANTIC_SEARCH_TOP_K, cosine_similarity_scores)]
    # Get the most relevant chunks
    most_relevant_chunks = [db_chunks[i] for i in most_relevant_chunks_indices]
    # Return the most relevant chunks
    return most_relevant_chunks

def cosine_similarity(embedding1: list[float], embedding2: list[float]):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def gen_groq_conversation_with_memory(groq_api_key: str):
    # Initialize memory
    memory = ConversationBufferWindowMemory(k=constants.CONVERSATION_MEMORY_LENGTH)
    # Initialize Groq Langchain chat object and conversation
    groq_chat = ChatGroq(
        temperature=constants.TEMPERATURE,
        groq_api_key=groq_api_key, 
        model_name=constants.MODEL_NAME
    )
    conversation = ConversationChain(
        llm=groq_chat,
        memory=memory
    )
    return conversation
