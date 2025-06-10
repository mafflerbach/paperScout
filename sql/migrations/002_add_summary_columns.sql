-- Add new summary columns to paper_sections table
ALTER TABLE paper_sections ADD COLUMN IF NOT EXISTS summary_academic TEXT;
ALTER TABLE paper_sections ADD COLUMN IF NOT EXISTS summary_business TEXT;
ALTER TABLE paper_sections ADD COLUMN IF NOT EXISTS summary_social TEXT;
ALTER TABLE paper_sections ADD COLUMN IF NOT EXISTS summary_educational TEXT;
ALTER TABLE paper_sections ADD COLUMN IF NOT EXISTS refinement_status VARCHAR(20) DEFAULT 'pending';
ALTER TABLE paper_sections ADD COLUMN IF NOT EXISTS refined_at TIMESTAMP WITH TIME ZONE; 