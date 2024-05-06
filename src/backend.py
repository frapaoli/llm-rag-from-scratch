import src.constants as constants
import src.utils as utils
import psycopg2.extras
from langchain_core.prompts.prompt import PromptTemplate

def ingest_docs(docs_dir_path: str, openai_api_key: str):

    # TODO (DB should be created if not exists yet)
    # Setup the DB
    # utils.setup_db()

    # Connect to the DB
    conn, cur = utils.connect_to_db()

    # Enable the usage of UUIDs
    psycopg2.extras.register_uuid()

    # Get the chunks and embeddings from the documents
    docs_names, docs_chunks, docs_embeddings = utils.get_chunks_embeddings_from_docs(docs_dir_path, openai_api_key)

    # Create DB table if not existing yet
    cur.execute(constants.DB_CREATE_TABLE_SQL_COMMAND)

    # Insert processed documents into the DB
    utils.insert_docs_into_db(docs_names, docs_chunks, docs_embeddings, conn, cur)

    # Close connection to the DB
    cur.close(), conn.close()

def ask_question(question: str, groq_api_key: str, openai_api_key: str):

    # Connect to the DB
    conn, cur = utils.connect_to_db()

    # Get embeddings and chunks from DB
    db_embeddings, db_chunks = utils.get_embeddings_chunks_from_db(cur)

    # Close connection to the DB
    cur.close(), conn.close()

    # Split the question into chunks
    question_chunks = utils.split_text(question)

    # Get embeddings of question chunks
    question_embeddings = [utils.get_text_embedding(chunk, constants.OPENAI_EMBEDDING_MODEL, openai_api_key) for chunk in question_chunks]

    # Get most relevant chunks for the asked question
    most_relevant_chunks = utils.get_most_relevant_chunks(question_embeddings, db_embeddings, db_chunks)

    # Generate conversation with memory
    conversation = utils.gen_groq_conversation_with_memory(groq_api_key)
    
    # Generate prompt from template, including the user question and the most relevant chunks to answer it
    prompt = PromptTemplate.from_template(constants.USER_QUESTION_PROMPT_TEMPLATE)
    prompt = prompt.format(
        context="\n".join(most_relevant_chunks),
        question=question
    )

    # Ask question (including relevant document chunks as context) to the LLM
    answer = conversation.invoke(prompt)
    return answer['response']

    # TODO investigate how can I use the embedding model of OpenAI without providing the API key
