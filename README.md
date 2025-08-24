# ChatWhiz 💬⚡

ChatWhiz is a lightweight, modular chat application framework designed for building and running AI-assisted chat systems. It combines traditional information retrieval with modern embeddings, allowing you to create a fast, extensible, and secure chat pipeline. The project comes with monitoring utilities, modular components for embedding and retrieval, and a simple web-based frontend.


## Features

* **Modular Architecture**: Organized into clearly separated modules (`embedder`, `retriever`, `vector_store`, etc.), making it easy to extend and customize.
* **Hybrid Retrieval**: Supports both **BM25** (lexical) and **vector-based** search for powerful and efficient query handling.
* **Encryption Support**: Built-in encryption utilities ensure chat data is stored securely.
* **Progress Monitoring**: Includes scripts like `monitor_progress.py` and `check_state.py` for inspecting indexing and runtime state.
* **Web API & Frontend**: A lightweight API (`api/server.py`) with static assets (`index.html`, `app.js`) for quick deployment of a chat UI.
* **Configurable**: All key parameters are managed through `config.yaml` and `.env.template` for environment setup.
* **Extensible LLM Support**: Includes `llm.py` for integrating large language models into the chat pipeline.


## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/Kinglord1090/ChatWhiz.git
   cd ChatWhiz
   ```

2. Create a virtual environment and install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux/Mac
   venv\Scripts\activate      # On Windows

   pip install -r requirements.txt
   ```

3. Configure environment variables:

   * Copy `.env.template` to `.env` and fill in the required values (API keys, secrets, etc.).
   * Adjust `config.yaml` for retrieval settings, paths, or model parameters.


## Usage

1. Start the server:

   ```bash
   python start_server.py
   ```

2. Open your browser and navigate to:

   ```
   http://localhost:5000
   ```

   This will launch the ChatWhiz web interface.

3. Optional utilities:

   * Check the indexing state:

     ```bash
     python check_state.py
     ```
   * Monitor progress:

     ```bash
     python monitor_progress.py
     ```


## Project Structure

```
ChatWhiz/
├── api/                # API server & static frontend
│   ├── server.py
│   └── static/
│       ├── index.html
│       └── app.js
├── modules/            # Core modules (embedding, retrieval, encryption, etc.)
├── data/               # Storage for chats, indexes, caches, and processed data
├── config.yaml         # Main configuration file
├── .env.template       # Environment variable template
├── requirements.txt    # Dependencies
├── start_server.py     # Main entrypoint
├── check_state.py      # Utility script
├── monitor_progress.py # Utility script
└── README.md
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

* Inspired by **retrieval-augmented generation (RAG)** systems.
* Uses **BM25** and vector search techniques for hybrid retrieval.
* Thanks to the open-source community for libraries enabling embeddings, encryption, and retrieval backends.

