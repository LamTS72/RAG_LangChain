## This is RAG System using Mistral-7B-Instruct-v0.2 which was quantized to 4-bit. In this system RAG also was deployed API with FastAPI
```
RAG_LangChain/
├── README.md
├── .gitignore
├── requirements.txt
├── src/
│   ├── app.py
│   ├── base/
│   │   ├── __init__.py
│   │   └── llm_model.py
|   |   └── utils.py
│   └── rag/
│       ├── __init__.py
│       └── file_loader.py
│       └── helper.py
│       └── rag_llm.py
│       └── vector_db.py
│       └── main.py
└── docs/
    └── file.pdf
```
## Run
```bash
uvicorn src.app:app --host "0.0.0.0" --port 8000 --reload
