
# 🚀 Stateful AI for Low-Bandwidth Networks (Patent Backend)

This repository contains the backend implementation for my patent, "System and Method for Stateful, Contextual AI Communication over Low-Bandwidth Networks" (Filed: FR2511116).

It's a complete, hybrid AI backend that allows for complex, stateful conversations (RAG, document analysis, web search) over low-bandwidth protocols like **SMS**.

## 🌟 Project Goal

The core innovation is to provide advanced AI capabilities (like RAG and VLM analysis) to users without high-speed internet.

The system works by:

1.  Receiving a query via a low-bandwidth channel (e.g., an SMS via Android Automate).
2.  Using this backend (`main-backend.py`) to perform all heavy lifting: RAG, document extraction, history retrieval, and context augmentation.
3.  Sending the compiled context + prompt to a dedicated LLM generation server (`api.py`).
4.  Receiving the final answer and sending it back to the user via SMS.

This architecture decouples perception/RAG (local) from generation (server) and enables stateful conversations over a stateless protocol.

## 🛠️ Key Features

  * **Hybrid Architecture:** The system is split into two main services:
    1.  **Main Backend (`main-backend.py`):** The "brain." It runs RAG, embedding, reranking, and document extraction. It manages all application logic.
    2.  **VLM/Generation Server (`api.py`):** A decoupled Flask server that hosts the primary generation model (e.g., Mistral Small). This `main-backend` calls this server *only* for the final text generation, separating responsibilities.
  * **SMS Integration:** Includes a `/api/sms_query` endpoint designed to receive requests from apps like **Automate (Android)**, manage conversation history per-user (by sender number), and return a simple text reply.
  * **Stateful SMS History:** Uses **ChromaDB** to create a persistent, vector-searched conversation history for *each* SMS user, allowing for follow-up questions.
  * **Advanced RAG Pipeline:**
      * **Document Extraction:** Uses a local VLM (**SmolDocling**) to extract text from images and PDFs.
      * **Vector Storage:** Uses **ChromaDB** for RAG document collection.
      * **Embedding:** Uses `ibm-granite/granite-embedding-278m-multilingual`.
      * **Reranking:** Uses `ibm-research/re2g-reranker-nq` to refine RAG results.
  * **Context Augmentation:** Can dynamically pull context from **Wikipedia** in addition to the RAG database and conversation history.

## 🚀 Getting Started

Follow these steps to configure and run the project.

### Prerequisites

  * Python 3.x
  * Git
  * Poppler (for `pdf2image` functionality)
  * For Apple Silicon (MLX models): `pip install mlx-vlm`

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/eauchs/patent-low-bandwidth-ai.git
    cd patent-low-bandwidth-ai
    ```

2.  **Create & Activate Virtual Environments (Recommended):**

    ```bash
    # For the main backend
    python3 -m venv venv_main
    source venv_main/bin/activate
    pip install -r requirements.txt # (Note: You need to create this file)

    # For the VLM API server
    python3 -m venv venv_api
    source venv_api/bin/activate
    pip install -r api/requirements.txt # (Note: You need to create this file)
    ```

3.  **Configuration:**
    Create a `.env` file in the root directory for `main-backend.py` and another in the `api/` directory for `api.py`.

    **For VLM Server (`./api/.env`):**

    ```
    VLM_MODEL_PATH="mlx-community/Mistral-Small-3.1-24B-Instruct-2503-3bit"
    VLM_SERVER_HOST="0.0.0.0"
    VLM_SERVER_PORT=5001
    ```

    **For Main Backend (`./.env`):**

    ```
    # Paths & Models
    CHROMA_DB_PATH="chromadb_base"
    EMBEDDING_MODEL_NAME="ibm-granite/granite-embedding-278m-multilingual"
    RERANKER_MODEL_NAME="ibm-research/re2g-reranker-nq"
    SMOL_VLM_MODEL_PATH="ds4sd/SmolDocling-256M-preview-mlx-bf16"

    # Endpoint for the VLM/Text server
    TEXT_GENERATION_SERVER_ENDPOINT="http://localhost:5001/api/vlm_generate"

    # RAG & Context Params
    WIKIPEDIA_LANG="fr"
    RAG_N_RESULTS=15
    RAG_TOP_K_RERANKED=10
    SMS_HISTORY_RAG_N_RESULTS=10
    ```

### Execution

1.  **Start the VLM / Text Generation Server:**

    ```bash
    source venv_api/bin/activate
    python api.py
    ```

    (The server will start on `http://0.0.0.0:5001`)

2.  **Start the Main Backend Server:**

    ```bash
    source venv_main/bin/activate
    python main-backend.py
    ```

    (The main application will start on `http://0.0.0.0:500`)

## ⚙️ API Usage

### `/api/generate` (Main Endpoint)

Send a POST request with JSON payload for RAG/Document-based queries.

```json
{
    "conversations": [
        {
            "role": "user",
            "content": [
                { "type": "text", "text": "What is the content of this image?" }
            ]
        }
    ],
    "file_data": "...", // base64 encoded image
    "file_name": "image.png",
    "options": {
        "document": true,
        "rag": false,
        "wikipedia": false
    }
}
```

### `/api/sms_query` (Patent Endpoint)

Send a `x-www-form-urlencoded` POST request (simulating an Android Automate webhook).

  * `sender`: The phone number (e.g., "+33612345678")
  * `message`: The user's text query (e.g., "What was the last thing I asked you?")

The backend will automatically find this user's vector history, perform RAG, get a response, and return it as simple JSON.

```json
{
    "reply": "You previously asked me about the contents of a PDF document."
}
```

