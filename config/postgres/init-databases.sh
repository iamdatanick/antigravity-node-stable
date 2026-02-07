#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE argo;
    CREATE DATABASE keycloak;
    CREATE DATABASE marquez;
    GRANT ALL PRIVILEGES ON DATABASE argo TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE keycloak TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE marquez TO $POSTGRES_USER;

    -- Marquez dev config expects user 'marquez' with password 'marquez'
    CREATE USER marquez WITH PASSWORD 'marquez';
    GRANT ALL PRIVILEGES ON DATABASE marquez TO marquez;
    ALTER DATABASE marquez OWNER TO marquez;
EOSQL
