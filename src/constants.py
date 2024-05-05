### LLM ###

# Name of the LLM to be used
MODEL_NAME = "llama3-70b-8192"
# Number of messages to be stored by the conversational LLM
CONVERSATION_MEMORY_LENGTH = 5
# Temperature of the LLM outputs
TEMPERATURE = 0
# User question prompt template
USER_QUESTION_PROMPT_TEMPLATE = """
You are a question-answering AI assistant. You have to answer a user question based on the provided context.

Context: {context}

Question: {question}

Answer:"""

### DOCUMENTS INGEST ###

# Directory path of documents to be ingested
DOCS_DIR_PATH = "./docs"
# Chunk size for the text splitter
TEXT_SPLITTER_CHUNK_SIZE = 512
# Chunk overlap for the text splitter
TEXT_SPLITTER_CHUNK_OVERLAP = 128
# OpenAI embedding model
OPENAI_EMBEDDING_MODEL="text-embedding-3-small"
# Embeddings length
EMBEDDINGS_LENGTH=1536

### USER MENU ###

# User menu request
USER_MENU_REQUEST = """
Insert an option number:
1. Ingest documents
2. Ask a question
3. Exit

Option number: """
# Mapping menu selections to user options 
MENU_SELECTION_TO_USER_OPTION_MAPPING = {
    '1': 'ingest_documents',
    '2': 'ask_question',
    '3': 'exit'
}

### EMBEDDINGS DATABASE ###

# Postgres host
POSTGRES_HOST="localhost"
# Postgres port
POSTGRES_PORT="5433"
# Postgres user
POSTGRES_USER="admin"
# Postgres password
POSTGRES_PASSWORD="root"
# Postgres database
POSTGRES_DATABASE="docs_chunks_embeddings"
# SQL command to create the table
DB_CREATE_TABLE_SQL_COMMAND = f"""
CREATE TABLE IF NOT EXISTS embedding (
    id uuid PRIMARY KEY,
    embedding real[],
    chunk varchar({TEXT_SPLITTER_CHUNK_SIZE})
)
"""
# SQL command to insert tuples in the table
DB_INSERT_TUPLE_SQL_COMMAND = """
INSERT INTO embedding (id, embedding, chunk) VALUES (%s, %s, %s)
"""
# SQL command to check if a tuple exists
DB_CHECK_TUPLE_EXISTS_SQL_COMMAND = """
SELECT * FROM embedding WHERE embedding = %s AND chunk = %s
"""

### RAG ###

# Number of most relevant chunks to be retrieved for the RAG
SEMANTIC_SEARCH_TOP_K = 3
