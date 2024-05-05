import os
from dotenv import load_dotenv, find_dotenv
import src.constants as constants
import src.menu as menu
import src.backend as backend

if __name__ == "__main__":
    
    # Load the environment variables
    load_dotenv(find_dotenv())

    # Get Groq and OpenAI API key from environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Get user option from menu
    user_option = menu.menu()

    # OPTION 1: Ingest documents
    if user_option == 'ingest_documents':
        backend.ingest_docs(constants.DOCS_DIR_PATH, openai_api_key)

    # OPTION 2: Ask question
    elif user_option == 'ask_question':
        question = input("Enter a question about the documents: ")
        answer = backend.ask_question(question, groq_api_key, openai_api_key)
        print(f"\nAnswer: {answer}\n")

    # OPTION 3: Exit
    elif user_option == 'exit':
        exit()
