-- PostgreSQL initialisation script for QB Risk Infra
-- Runs automatically when the postgres container starts for the first time.

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges to the application user
GRANT ALL PRIVILEGES ON DATABASE qb_risk TO postgres;

-- Create a read-only role for analytics queries (optional)
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'qb_readonly') THEN
        CREATE ROLE qb_readonly LOGIN PASSWORD 'readonly_changeme';
    END IF;
END $$;

GRANT CONNECT ON DATABASE qb_risk TO qb_readonly;
GRANT USAGE ON SCHEMA public TO qb_readonly;
