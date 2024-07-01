import src.constants as constants
import src.utils as utils
from langchain_core.prompts.prompt import PromptTemplate
from pinecone import Pinecone
import uuid

def ingest_docs(docs_dir_path: str, openai_api_key: str, pinecone_api_key: str):

    # Get the chunks and embeddings from the documents
    docs_names, docs_chunks, docs_embeddings = utils.get_chunks_embeddings_from_docs(docs_dir_path, openai_api_key)

    # TODO: use design pattern to avoid if-else structure and enhance modularity for future vector DB options
    # Ingest documents, storing its chunks and embeddings into the DB
    if constants.VECTOR_DB_OPTION == "postgres":
        utils.ingest_docs_postgres_db(docs_names, docs_chunks, docs_embeddings)
    elif constants.VECTOR_DB_OPTION == "pinecone":
        utils.ingest_docs_pinecone_db(docs_names, docs_chunks, docs_embeddings, pinecone_api_key)
    else:
        raise ValueError(f"Invalid vector DB option: {constants.VECTOR_DB_OPTION}")

def ask_question(question: str, groq_api_key: str, openai_api_key: str, pinecone_api_key: str):

    # Connect to the DB
    conn, cur = utils.connect_to_db()

    # Get embeddings and chunks from DB
    db_embeddings, db_chunks = utils.get_embeddings_chunks_from_db(cur)

    # Close connection to the DB
    cur.close(), conn.close()

    # Split the question into chunks
    # question_chunks = utils.split_text(question)

    # Get embeddings of question chunks
    # question_embeddings = [utils.get_text_embedding(chunk, constants.OPENAI_EMBEDDING_MODEL, openai_api_key) for chunk in question_chunks]
    question_embedding = utils.get_text_embedding(question, constants.OPENAI_EMBEDDING_MODEL, openai_api_key)

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index("llm-rag-from-scratch-index")
    query_response = index.query(
        namespace="llm-rag-from-scratch-namespace",
        vector=question_embedding,
        top_k=constants.SEMANTIC_SEARCH_TOP_K,
        include_values=False,
        include_metadata=True
    )

    print(query_response)
    print("\n\n")

    # Get most relevant chunks for the asked question
    most_relevant_chunks = utils.get_most_relevant_chunks(question_embedding, db_embeddings, db_chunks)

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
