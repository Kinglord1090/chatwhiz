# ChatWhiz v2 - Advanced Chat Search System

An intelligent semantic search system for chat exports with support for WhatsApp, JSON, and CSV formats. Features include semantic search, BM25 keyword search, hybrid search, and an AI-powered Q&A assistant.

## 🚀 Features

### Search Capabilities
- **Semantic Search**: Find messages by meaning using state-of-the-art embeddings
- **BM25 Search**: Traditional keyword-based search
- **Hybrid Search**: Combines semantic and keyword search for best results
- **AI Assistant**: Built-in Q&A system that analyzes and answers questions about your chats

### Advanced Features
- **Real-time Progress Tracking**: Detailed progress updates with decimal precision
- **Persistent State**: Automatically resumes interrupted indexing after restart
- **Batch Processing**: Efficient handling of large chat files
- **Smart Caching**: Reuses embeddings to speed up re-indexing
- **Clean Shutdown**: Gracefully handles interruptions with state preservation

### File Support
- WhatsApp chat exports (.txt)
- JSON format chat files
- CSV format chat files
- Automatic format detection
- Encryption detection

## 📋 Requirements

- Python 3.8 or higher
- 4GB+ RAM recommended
- CUDA-capable GPU (optional, for faster processing)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/ChatWhiz_v2.git
cd ChatWhiz_v2
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure settings** (optional)
```bash
# Copy the template
cp .env.template .env

# Edit config.yaml to customize settings
```

## 🚀 Quick Start

1. **Start the server**
```bash
python start_server.py
```

2. **Open the web interface**
   - Navigate to http://localhost:8000 in your browser
   - The modern UI will load automatically

3. **Upload and index your chats**
   - Click the "Upload & Index" tab
   - Drag and drop or select your chat files
   - Click "Process & Index" to start

4. **Search your chats**
   - Switch to the "Search" tab
   - Enter your query
   - Choose search mode (Semantic/BM25/Hybrid)
   - View results and use the AI Assistant for insights

## 🔧 Configuration

### config.yaml Options

```yaml
# Embedding model configuration
embedding_model: 'hkunlp/instructor-large'
device: 'auto'  # 'auto', 'cpu', or 'cuda'

# Search settings
retrieval_mode: 'semantic'
top_k: 5
similarity_threshold: 0.3

# Directory settings
data_dir: 'data'
vectorstore_dir: 'data/vectorstore'
bm25_dir: 'data/bm25'
cache_dir: 'data/cache'
```

### Environment Variables (.env)

```bash
# Optional: External LLM configuration
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## 🎯 Usage Examples

### Search Modes

**Semantic Search**: Best for conceptual queries
- "discussions about vacation plans"
- "messages expressing happiness"
- "technical problems mentioned"

**BM25 Search**: Best for exact terms
- Specific names or phrases
- Exact keyword matches
- Technical terms

**Hybrid Search**: Balanced approach
- Combines both methods
- Best overall accuracy

### AI Assistant

Ask questions like:
- "Who talked most about travel?"
- "What were the main topics discussed?"
- "How many messages mention food?"
- "When was the project deadline mentioned?"

## 📁 Project Structure

```
ChatWhiz_v2/
├── api/
│   ├── server.py          # FastAPI backend
│   └── static/
│       ├── index.html     # Web interface
│       └── app.js         # Frontend logic
├── modules/
│   ├── embedder.py        # Embedding generation
│   ├── vector_store.py    # FAISS vector storage
│   ├── bm25_store.py      # BM25 index
│   ├── retriever.py       # Search implementation
│   ├── loader.py          # File parsing
│   ├── llm.py             # LLM integration
│   └── encryptor.py       # Encryption detection
├── data/                  # Generated data (gitignored)
├── config.yaml            # Configuration
├── requirements.txt       # Dependencies
└── start_server.py        # Server launcher
```

## 🔍 API Endpoints

- `GET /` - Web interface
- `GET /api/status` - System status
- `POST /api/search` - Search messages
- `POST /api/rag` - AI Assistant queries
- `POST /api/upload` - Upload files
- `GET /api/task/{id}` - Check task progress

## 🐛 Troubleshooting

### Common Issues

**"No module named 'modules'"**
- Run from the project root directory
- Use `python start_server.py` instead of running server.py directly

**GPU not detected**
- Install CUDA toolkit and PyTorch with CUDA support
- Set `device: 'cpu'` in config.yaml to use CPU only

**Indexing interrupted**
- Simply restart the server - it will resume automatically
- Check `data/indexing_state.json` for saved state

**Search returns no results**
- Ensure files are properly indexed (check Analytics tab)
- Try different search modes
- Lower the similarity threshold

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [Instructor Embeddings](https://instructor-embedding.github.io/) for semantic search
- [FAISS](https://github.com/facebookresearch/faiss) for vector storage
- [FastAPI](https://fastapi.tiangolo.com/) for the backend framework
- [Tailwind CSS](https://tailwindcss.com/) for the UI design

## 📞 Support

For issues and questions:
- Open an issue on GitHub
- Check existing issues for solutions
- Provide logs and error messages when reporting bugs

---

**Note**: This is a local, privacy-focused application. All data processing happens on your machine, and no data is sent to external servers unless you configure external LLM providers.
