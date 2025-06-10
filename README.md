# 📚 PaperScout

**AI-Powered Academic Paper Analysis and Discovery System**

PaperScout is a comprehensive platform for discovering, analyzing, and extracting insights from academic papers using AI. It combines semantic search, automated content analysis, and intelligent metadata extraction to help researchers, content creators, and professionals stay up-to-date with the latest academic developments.

## ✨ Features

### 🔍 **Intelligent Paper Discovery**
- **Semantic Search**: Vector-based search using embeddings for finding conceptually similar papers
- **Traditional Text Search**: Full-text search across paper titles, abstracts, and content
- **Smart Filtering**: Filter by categories, difficulty level, business relevance, and more
- **ArXiv Integration**: Direct support for ArXiv papers with automated metadata extraction

### 🤖 **AI-Powered Analysis**
- **Automatic Content Extraction**: PDF text extraction and structured content parsing
- **LLM-Based Metadata Generation**: AI-generated topics, techniques, and applications
- **Business Relevance Scoring**: Automated scoring for commercial applicability (1-10 scale)
- **Difficulty Assessment**: Automatic classification as beginner, intermediate, or advanced
- **Implementation Feasibility**: Assessment of how practical the research is to implement

### 📊 **Content Management**
- **Paper Library**: Organized storage of papers with rich metadata
- **Section-Based Analysis**: Papers broken down into sections (abstract, methodology, results, etc.)
- **Tag System**: Flexible tagging with confidence scores
- **Content Creation Tracking**: Mark papers for content creation and track notes

### 🔧 **Technical Capabilities**
- **Vector Database**: PostgreSQL with pgvector extension for semantic search
- **PDF Processing**: Robust PDF text extraction using PyMuPDF and pdfplumber
- **Local LLM Integration**: Works with LM Studio for privacy-focused AI processing
- **Scalable Architecture**: Docker-based microservices with optional worker processes
- **Modern Frontend**: React-based UI with Tailwind CSS styling

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│                 │    │                 │    │                 │
│   React Frontend│◄───┤   FastAPI       │◄───┤  PostgreSQL     │
│   (Port 5173)   │    │   Backend       │    │  + pgvector     │
│                 │    │   (Port 8000)   │    │  (Port 5432)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │                 │
                       │   LM Studio     │
                       │   (Port 1234)   │
                       │                 │
                       └─────────────────┘
```

### Core Components

- **Frontend**: React + Vite + Tailwind CSS for modern, responsive UI
- **API**: FastAPI backend with async support and automatic documentation
- **Database**: PostgreSQL with pgvector extension for vector similarity search
- **AI Engine**: Integration with local LLM via LM Studio for privacy-focused processing
- **Worker**: Optional background processing for batch operations
- **Storage**: Local PDF storage with organized file management

## 🚀 Quick Start

### Prerequisites

- **Docker & Docker Compose** installed
- **LM Studio** running locally with a model loaded
- **Python 3.9+** (for optional local development)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd paperScout
   ```

2. **Start LM Studio**
   - Open LM Studio
   - Load a compatible model (e.g., llama-3.2-3b-instruct, mistral-7B)
   - Start the local server on port 1234
   - Verify: `curl http://localhost:1234/v1/models`

3. **Configure environment**
   ```bash
   # Copy and edit the environment variables in docker-compose.yml
   # Update the LLM_MODEL name to match your loaded model
   ```

4. **Start the application**
   ```bash
   # Start core services
   docker-compose up -d

   # Or with development tools (includes pgAdmin)
   docker-compose --profile development up -d

   # With background worker (for batch processing)
   docker-compose --profile worker up -d
   ```

5. **Verify installation**
   ```bash
   # Check API health
   curl http://localhost:8000/health

   # Access the application
   open http://localhost:5173
   ```

### Testing with Sample Data

Run the included test to verify everything works:

```bash
# Test with sample papers
python test_paperscout.py
```

This will:
- Add sample papers to the database
- Download and process PDFs
- Generate AI metadata
- Test search functionality

## 📖 Usage

### Adding Papers

1. **Via API**:
   ```bash
   curl -X POST http://localhost:8000/api/papers/ \
     -H "Content-Type: application/json" \
     -d '{
       "title": "Paper Title",
       "url": "https://arxiv.org/pdf/2301.00000.pdf",
       "arxiv_id": "2301.00000"
     }'
   ```

2. **Bulk Import**:
   ```bash
   # Use the test_papers.json format for batch imports
   curl -X POST http://localhost:8000/api/papers/bulk \
     -H "Content-Type: application/json" \
     -d @test_papers.json
   ```

### Processing Papers

```bash
# Full processing (download + extract + analyze)
curl -X POST http://localhost:8000/api/processing/papers/{paper_id}/full

# Check processing status
curl http://localhost:8000/api/papers/{paper_id}
```

### Searching Papers

```bash
# Semantic search
curl "http://localhost:8000/api/search/?q=transformer%20attention&limit=5"

# Filter by business relevance
curl "http://localhost:8000/api/papers/?business_relevance_min=7"

# Filter by category
curl "http://localhost:8000/api/papers/?categories=machine%20learning"
```

## 🔧 Configuration

### Environment Variables

Key configuration options in `docker-compose.yml`:

```yaml
# Database
DATABASE_URL: postgresql://scout:password@postgres:5432/paperscout

# LLM Integration
LLM_ENDPOINT: http://host.docker.internal:1234/v1/chat/completions
LLM_MODEL: llama-3.2-3b-instruct
EMBEDDING_MODEL: sentence-transformers/all-MiniLM-L6-v2

# Storage
PDF_STORAGE_PATH: /app/pdfs
```

### LM Studio Models

Recommended models:
- **llama-3.2-3b-instruct**: Fast, good for metadata extraction
- **mistral-7b-instruct**: Balanced performance and quality
- **llama-3.1-8b-instruct**: Higher quality analysis

## 🗄️ Database Schema

### Core Tables

- **papers**: Main paper metadata, processing status, AI-generated insights
- **paper_sections**: Sectioned content with embeddings for semantic search
- **tags**: Normalized tagging system with categories
- **paper_tags**: Many-to-many relationship with confidence scores
- **search_queries**: Query tracking and feedback collection

### Key Features

- **Vector Search**: pgvector extension for semantic similarity
- **Full-Text Search**: PostgreSQL's built-in text search
- **Flexible Tagging**: Multi-category tags with confidence scores
- **Processing Pipeline**: Status tracking from download to analysis

## 🛠️ Development

### Local Development Setup

1. **Backend Development**:
   ```bash
   cd api
   pip install -r requirements.txt
   uvicorn src.app.main:app --reload --port 8000
   ```

2. **Frontend Development**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

3. **Database Access**:
   ```bash
   # Via pgAdmin (if running with development profile)
   open http://localhost:8080

   # Via command line
   docker-compose exec postgres psql -U scout -d paperscout
   ```

### API Documentation

Access the interactive API documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Tech Stack

**Backend**:
- FastAPI (Python web framework)
- SQLAlchemy (ORM)
- PostgreSQL + pgvector (database)
- sentence-transformers (embeddings)
- PyMuPDF + pdfplumber (PDF processing)

**Frontend**:
- React 18
- Vite (build tool)
- Tailwind CSS (styling)
- Lucide React (icons)
- Axios (HTTP client)

**Infrastructure**:
- Docker & Docker Compose
- LM Studio (local LLM server)
- Redis (optional, for job queues)

## 📊 Monitoring & Management

### Health Checks

```bash
# API health
curl http://localhost:8000/health

# Database health
docker-compose exec postgres pg_isready -U scout

# View logs
docker-compose logs api
docker-compose logs frontend
```

### Database Management

```bash
# Connect to database
docker-compose exec postgres psql -U scout -d paperscout

# Backup database
docker-compose exec postgres pg_dump -U scout paperscout > backup.sql

# View paper processing stats
docker-compose exec postgres psql -U scout -d paperscout -c "
  SELECT processing_status, COUNT(*) 
  FROM papers 
  GROUP BY processing_status;
"
```

## 🔒 Security Considerations

- **Database**: Change default passwords in production
- **LLM Processing**: Runs locally via LM Studio (no data sent to external APIs)
- **File Storage**: PDFs stored locally with controlled access
- **API**: Add authentication for production deployment

## 🚀 Deployment

### Production Deployment

1. **Update configurations**:
   - Change database passwords
   - Configure proper PDF storage paths
   - Set up SSL/TLS certificates
   - Configure domain names

2. **Scale services**:
   ```bash
   # Scale worker processes
   docker-compose --profile worker up -d --scale worker=3
   
   # Use external database
   # Update DATABASE_URL to point to managed PostgreSQL
   ```

3. **Monitoring**:
   - Set up log aggregation
   - Configure health check monitoring
   - Set up backup procedures

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📋 Roadmap

- [ ] **ArXiv Integration**: Automated paper discovery and ingestion
- [ ] **Advanced Search**: Filters by author, publication date, citation count
- [ ] **Content Creation**: LinkedIn/Twitter post generation from papers
- [ ] **Research Insights**: Trend analysis and research landscape mapping
- [ ] **Browser Extension**: One-click paper saving from web

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **pgvector**: PostgreSQL extension for vector similarity search
- **LM Studio**: Local LLM server for privacy-focused AI processing
- **FastAPI**: Modern Python web framework
- **React**: Frontend library for building user interfaces

## 📞 Support

For support, please:
1. Check the [Testing.md](Testing.md) file for troubleshooting
2. Review the API documentation at http://localhost:8000/docs
3. Open an issue on GitHub
4. Contact the development team

---

**Happy paper scouting! 📚🔍** 