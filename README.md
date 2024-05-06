# Installation steps

- Install Docker
- Install PostgreSQL
- `python3 -m venv .venv` within project's root directory
- `source .venv/bin/activate` within project's root directory
- `sh setup.sh` within project's root directory
- Create a file `.env` within project's root directory and insert in it your OpenAI and Groq API keys as strings, naming them OPENAI_API_KEY and GROQ_API_KEY, respectively

# PostgreSQL setup steps
- `docker-compose up -d` within project's root directory
- Open the browser and go to the URL `http://localhost:5050/`
- Login with following credentials:
    - Username: admin@admin.com
    - Password: root
- On the left bar, right-click on `Servers` and select `Register > Server...`
- In the General tab, insert:
    - Name: llm-rag-from-scratch-server
- In the Cnnection tab, insert:
    - Host name/address: postgres
    - Port: 5432
    - Maintenance database: postgres
    - Username: admin
    - Password: root
- Click Save
- On the left bar, expand llm-rag-from-scratch-server, right-click on `Databases` and select `Create > Database...`
- In the General tab, insert:
    - Name: docs_chunks_embeddings
- Click Save

# Usage steps
- Insert in `docs` folder the PDF files you want the LLM to ingest
- `python3 app.py` within project's root directory
