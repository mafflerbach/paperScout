# Database Management

This directory contains SQL scripts for managing the PaperScout database.

## Reset and Initialize Database

To completely reset and reinitialize the database, follow these steps:

1. First, reset the database (this will delete all data):
```bash
psql -d paperscout -f sql/reset.sql
```

2. Then initialize the database schema:
```bash
psql -d paperscout -f sql/init.sql
```

3. After initializing, you can load test papers using:
```bash
curl -X POST http://localhost:8000/api/papers/batch \
  -H "Content-Type: application/json" \
  -d @test_papers.json
```

## Important Notes

- The reset script will drop all tables and functions
- The init script will:
  - Create all necessary tables
  - Set up indexes
  - Create the update timestamp trigger
  - Insert initial tag data
- Make sure your PostgreSQL server is running and the paperscout database exists
- The pgvector extension must be installed in your PostgreSQL instance

## Database Schema

The database consists of the following main tables:

- `papers`: Main table for paper metadata and content
- `paper_sections`: Stores paper sections and their summaries
- `tags`: Normalized tag data
- `paper_tags`: Junction table for paper-tag relationships
- `search_queries`: Tracks search history and feedback

Each table has appropriate indexes for performance optimization. 