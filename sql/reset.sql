-- Drop all tables
DROP TABLE IF EXISTS paper_sections CASCADE;
DROP TABLE IF EXISTS paper_tags CASCADE;
DROP TABLE IF EXISTS tags CASCADE;
DROP TABLE IF EXISTS search_queries CASCADE;
DROP TABLE IF EXISTS papers CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS update_updated_at_column CASCADE;

-- Drop extensions (optional, comment out if you want to keep the extension)
-- DROP EXTENSION IF EXISTS vector; 