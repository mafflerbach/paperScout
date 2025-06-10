# 🚀 PaperScout Testing Guide

## Prerequisites

1. **Docker & Docker Compose** installed
2. **LM Studio** running locally on port 1234 with a model loaded
3. **Python** for the test script (optional)

## Step 1: Setup Project Structure

```bash
# Run the setup commands from earlier
mkdir -p paperscout/{api/src/app,frontend/src,sql,pdfs,models}

# Create all the files we discussed
# (Copy the content from the artifacts into the respective files)
```

## Step 2: Start LM Studio

1. Open LM Studio
2. Load any decent model (e.g., llama-3.1-8B, mistral-7B)
3. Start the local server on port 1234
4. Verify it's running: `curl http://localhost:1234/v1/models`

## Step 3: Start PaperScout

```bash
cd paperscout

# Start the basic services
docker-compose up -d

# Or with development tools
docker-compose --profile development up -d

# Check if services are running
curl http://localhost:8000/health
```

## Step 4: Create Test Data

Save the test papers JSON as `test_papers.json` in your paperscout directory.

## Step 5: Run the Test

```bash
# Install httpx for the test script
pip install httpx

# Run the test
python test_paperscout.py
```

## What the Test Does

1. ✅ **Health Check** - Verifies API is running
2. 📄 **Create Papers** - Adds 3 famous papers to database
3. 🔄 **Process Paper** - Downloads PDF, extracts text, runs LLM analysis
4. 🔍 **Test Search** - Semantic and text search
5. 📊 **Get Stats** - Processing statistics
6. 📝 **Content Ready** - Papers ready for LinkedIn posts

## Expected Results

- Papers should download successfully
- Text extraction should work 
- LLM should generate topics, techniques, business relevance scores
- Search should find relevant papers
- You'll see which papers have high business relevance for content creation

## Troubleshooting

- **LM Studio not responding**: Check it's running on port 1234
- **PDF download fails**: Check internet connection and URL validity
- **Database errors**: Ensure PostgreSQL container is healthy
- **Slow processing**: Academic PDFs can be large, be patient

## Next Steps

Once testing works:
1. Feed it real ArXiv papers from your 6 categories
2. Tune the LLM prompts for better metadata extraction
3. Build the frontend for easier paper management
4. Set up automated ArXiv crawling

## Useful Commands

```bash
# Check logs
docker-compose logs api

# Access database
docker-compose exec postgres psql -U scout -d paperscout

# Manual paper processing
curl -X POST http://localhost:8000/api/processing/papers/1/full

# Search for papers
curl "http://localhost:8000/api/search/?q=transformer&limit=5"
```
