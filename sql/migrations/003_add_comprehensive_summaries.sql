-- Add comprehensive summary columns to papers table
ALTER TABLE papers ADD COLUMN IF NOT EXISTS comprehensive_summary TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS comprehensive_summary_academic TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS comprehensive_summary_business TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS comprehensive_summary_social TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS comprehensive_summary_educational TEXT;
ALTER TABLE papers ADD COLUMN IF NOT EXISTS comprehensive_summary_updated_at TIMESTAMP WITH TIME ZONE; 