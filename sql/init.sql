CREATE INDEX idx_content_posts_paper_id ON content_posts(paper_id);
CREATE INDEX idx_content_posts_platform ON content_posts(platform);
CREATE INDEX idx_content_posts_status ON content_posts(status);-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Papers table - main paper metadata
CREATE TABLE papers (
    id SERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    url TEXT NOT NULL UNIQUE,
    pdf_path TEXT,
    arxiv_id TEXT,
    authors TEXT[],
    abstract TEXT,
    published_date DATE,
    categories TEXT[],
    
    -- Processing status
    download_status VARCHAR(20) DEFAULT 'pending', -- pending, downloaded, failed
    processing_status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    
    -- Full text content
    full_text TEXT,
    
    -- LLM generated metadata
    main_topics TEXT[],
    techniques TEXT[],
    applications TEXT[],
    difficulty_level VARCHAR(20), -- beginner, intermediate, advanced
    implementation_feasibility VARCHAR(20), -- low, medium, high
    business_relevance_score INTEGER CHECK (business_relevance_score >= 1 AND business_relevance_score <= 10),
    
    -- Content creation tracking
    content_created BOOLEAN DEFAULT FALSE,
    content_note_path TEXT -- Path to Obsidian note/markdown file
);

-- Paper sections - split content with summaries
CREATE TABLE paper_sections (
    id SERIAL PRIMARY KEY,
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    section_type VARCHAR(50), -- abstract, introduction, methodology, results, conclusion, etc.
    section_title TEXT,
    content TEXT NOT NULL,
    summary TEXT,
    order_index INTEGER,
    
    -- Multi-style refined summaries
    summary_academic TEXT,    -- Technical precision for researchers
    summary_business TEXT,    -- ROI and practical applications focus
    summary_social TEXT,      -- Optimized for LinkedIn/Twitter posts
    summary_educational TEXT, -- Accessible for students/learners
    
    -- Refinement tracking
    refinement_status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    refined_at TIMESTAMP WITH TIME ZONE,
    
    -- Vector embedding for semantic search
    embedding vector(384), -- Adjust dimension based on your embedding model
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tags table - for normalized tagging
CREATE TABLE tags (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    category VARCHAR(50), -- topic, technique, application, etc.
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Paper tags junction table
CREATE TABLE paper_tags (
    paper_id INTEGER REFERENCES papers(id) ON DELETE CASCADE,
    tag_id INTEGER REFERENCES tags(id) ON DELETE CASCADE,
    confidence_score FLOAT CHECK (confidence_score >= 0 AND confidence_score <= 1),
    PRIMARY KEY (paper_id, tag_id)
);

-- Search and discovery tracking
CREATE TABLE search_queries (
    id SERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    query_vector vector(384),
    result_count INTEGER,
    user_feedback VARCHAR(20), -- relevant, irrelevant, partially_relevant
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_papers_categories ON papers USING GIN(categories);
CREATE INDEX idx_papers_main_topics ON papers USING GIN(main_topics);
CREATE INDEX idx_papers_processing_status ON papers(processing_status);
CREATE INDEX idx_papers_published_date ON papers(published_date DESC);
CREATE INDEX idx_papers_business_relevance ON papers(business_relevance_score DESC);
CREATE INDEX idx_papers_content_created ON papers(content_created);

CREATE INDEX idx_paper_sections_paper_id ON paper_sections(paper_id);
CREATE INDEX idx_paper_sections_type ON paper_sections(section_type);
CREATE INDEX idx_paper_sections_embedding ON paper_sections USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_paper_sections_refinement_status ON paper_sections(refinement_status);

CREATE INDEX idx_tags_name ON tags(name);
CREATE INDEX idx_tags_category ON tags(category);

CREATE INDEX idx_content_posts_paper_id ON content_posts(paper_id);
CREATE INDEX idx_content_posts_platform ON content_posts(platform);
CREATE INDEX idx_content_posts_status ON content_posts(status);

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_papers_updated_at BEFORE UPDATE ON papers
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Insert some common tags to get started
INSERT INTO tags (name, category) VALUES
    -- Topics
    ('machine learning', 'topic'),
    ('deep learning', 'topic'),
    ('computer vision', 'topic'),
    ('natural language processing', 'topic'),
    ('reinforcement learning', 'topic'),
    ('multiagent systems', 'topic'),
    ('robotics', 'topic'),
    ('optimization', 'topic'),
    ('graph neural networks', 'topic'),
    ('federated learning', 'topic'),
    
    -- Techniques
    ('transformer', 'technique'),
    ('convolutional neural network', 'technique'),
    ('generative adversarial network', 'technique'),
    ('attention mechanism', 'technique'),
    ('transfer learning', 'technique'),
    ('few-shot learning', 'technique'),
    ('meta learning', 'technique'),
    ('ensemble methods', 'technique'),
    
    -- Applications
    ('autonomous driving', 'application'),
    ('medical imaging', 'application'),
    ('financial modeling', 'application'),
    ('recommender systems', 'application'),
    ('fraud detection', 'application'),
    ('sentiment analysis', 'application'),
    ('speech recognition', 'application'),
    ('drug discovery', 'application'),
    ('climate modeling', 'application'),
    ('algorithmic trading', 'application');
