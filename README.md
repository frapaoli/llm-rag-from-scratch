# Installation steps

- Install Docker
- Install PostgreSQL
- `python3 -m venv .venv` within project's root directory
- `source .venv/bin/activate` within project's root directory
- `sh setup.sh` within project's root directory
- `docker-compose up -d` within project's root directory
- Create a file `.env` within project's root directory and insert in it your OpenAI and Groq API keys as strings, naming them OPENAI_API_KEY and GROQ_API_KEY, respectively.

# Usage steps
- Insert in `docs` folder the PDF files you want the LLM to ingest.
- `python3 app.py` within project's root directory

