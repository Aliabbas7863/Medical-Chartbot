# Medical Chartbot

A Streamlit-based medical question-answering chatbot that responds using content from your uploaded medical PDF. The app is designed to answer questions from the document context and say **"I don't know"** when the answer is not present in the PDF.

## What it does

- Loads a medical PDF from the local `data/` folder
- Extracts and chunks text from the PDF
- Builds a local FAISS vector index for retrieval
- Answers user questions using only the retrieved document text
- Returns **"I don't know the answer based on the provided documents"** when no relevant text is found

## Main idea

The chatbot is not a general web search assistant. It is meant to answer questions that are **supported by the document you provided**.

Example:
- If the PDF explains what cancer is, the chatbot should answer that question.
- If the PDF does not contain the topic, the chatbot should reply that it does not know.

## Project structure

```text
Medical Chartbot/
├── README.md
└── .venv/
    ├── medical_chatbot.py
    ├── data/
    │   └── The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf
    ├── faiss_index/
    ├── .env
    └── other virtual environment files...
```

### Important files

- `.venv/medical_chatbot.py`  
  Main Streamlit application.

- `.venv/data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf`  
  The source PDF used for document-based answers.

- `.venv/faiss_index/`  
  Local FAISS index folder used for similarity search.

- `.venv/.env`  
  Environment variables such as `HF_TOKEN` if you choose to use hosted LangChain/HuggingFace features.

## Requirements

The app uses:

- `streamlit`
- `PyPDF2`
- `sentence-transformers`
- `faiss-cpu`
- `numpy`
- `transformers`
- `python-dotenv`

Some optional LangChain imports are also present, but the app now has a local PDF-based fallback path when LangChain is unavailable.

## How it works

1. The PDF is read from the local `data/` folder.
2. The text is split into overlapping chunks.
3. Each chunk is converted into embeddings using `sentence-transformers`.
4. A FAISS index is built for fast similarity search.
5. When you ask a question, the app searches for the most relevant chunks.
6. If matching text is found, that text is returned as the answer.
7. If no useful match is found, the app says it does not know.

## How to run

From the project folder:

```bash
/home/Abbas/Desktop/Medical Chartbot/.venv/bin/streamlit run .venv/medical_chatbot.py --server.port 8501 --server.headless true
```

Then open:

```text
http://localhost:8501
```

## Example questions

Try questions that are likely covered in the PDF:

- What is cancer?
- What are the symptoms of diabetes?
- What causes fever?
- How is a disease diagnosed?


## Screenshot
<img width="624" height="466" alt="charbot" src="https://github.com/user-attachments/assets/539fb4fd-c672-46e2-a6b4-c5c21118c7a3" />
<img width="624" height="424" alt="charbot 1" src="https://github.com/user-attachments/assets/34b1ea79-7530-4f47-aad2-f8a8a5d88edd" />
<img width="624" height="284" alt="chatbot 3" src="https://github.com/user-attachments/assets/0b63c1a1-e508-46e7-9d2e-f1dea084491e" />











## Notes

- The app is only as good as the text in the PDF.
- If the PDF does not contain the answer, the chatbot should not invent one.
- Better retrieval depends on how clearly the PDF text is extracted.

## Troubleshooting

### The app says "I don't know"

This usually means:
- The PDF does not contain the answer
- The question is phrased too differently from the text in the PDF
- The relevant text was not extracted cleanly from the PDF

### Port is already in use

If port 8501 is busy, stop other Streamlit processes or choose another port, for example 8502.

## Next improvements

Possible future improvements:

- Index multiple PDFs
- Improve chunking strategy
- Add answer summarization from retrieved context
- Add a document upload interface
- Add conversation memory for follow-up questions
