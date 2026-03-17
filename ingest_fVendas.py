"""
ingest_fVendas.py
Extrai fVendas do SQL Server e carrega no PostgreSQL (schema integralmix).
Projetado para rodar standalone ou como PythonOperator no Airflow.
"""

import logging
import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv(".env.ingest")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engines
# ---------------------------------------------------------------------------

def build_sqlserver_engine():
    host = os.environ["SQLSERVER_HOST"]
    db   = os.environ["SQLSERVER_DB"]
    user = os.environ["SQLSERVER_USER"]
    pw   = os.environ["SQLSERVER_PASS"]
    url  = f"mssql+pymssql://{user}:{pw}@{host}/{db}"
    return create_engine(url)


def build_postgres_engine():
    host = os.environ["POSTGRES_HOST"]
    port = os.environ.get("POSTGRES_PORT", "5432")
    db   = os.environ["POSTGRES_DB"]
    user = os.environ["POSTGRES_USER"]
    pw   = os.environ["POSTGRES_PASS"]
    url  = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"
    return create_engine(url)


# ---------------------------------------------------------------------------
# Passos
# ---------------------------------------------------------------------------

def ensure_schema(pg_engine) -> None:
    """Cria o schema integralmix no Postgres se não existir."""
    log.info("Verificando schema integralmix no PostgreSQL...")
    with pg_engine.begin() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS integralmix;"))
    log.info("Schema integralmix OK.")


def extract(ss_engine) -> pd.DataFrame:
    """Extrai todos os dados de fVendas do SQL Server."""
    log.info("Iniciando extração de fVendas no SQL Server...")
    df = pd.read_sql("SELECT * FROM dbo.INT_Vendas_VendaRealizada", con=ss_engine)
    log.info(f"Extração concluída: {len(df):,} linhas | {len(df.columns)} colunas.")
    return df


def load(df: pd.DataFrame, pg_engine) -> None:
    """
    Carrega o DataFrame no PostgreSQL.
    if_exists='replace' dropa e recria a tabela a cada execução
    — ideal para carga diária completa no MVP.
    """
    log.info("Iniciando carga no PostgreSQL (integralmix.fVendas)...")
    df.to_sql(
        name="fVendas",
        con=pg_engine,
        schema="integralmix",
        if_exists="replace",
        index=False,
        chunksize=1000,      # envia em lotes para não estourar memória
        method="multi",      # INSERT multi-row, muito mais rápido
    )
    log.info(f"Carga concluída: {len(df):,} linhas inseridas em integralmix.fVendas.")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run():
    log.info("=== Início da ingestão fVendas ===")
    try:
        ss_engine = build_sqlserver_engine()
        pg_engine = build_postgres_engine()

        ensure_schema(pg_engine)
        df = extract(ss_engine)
        load(df, pg_engine)

        log.info("=== Ingestão finalizada com sucesso ===")

    except KeyError as e:
        log.error(f"Variável de ambiente ausente: {e}. Verifique o arquivo .env.ingest.")
        raise
    except Exception as e:
        log.error(f"Falha na ingestão: {e}", exc_info=True)
        raise
    finally:
        # Garante fechamento dos pools de conexão
        try:
            ss_engine.dispose()
            pg_engine.dispose()
        except Exception:
            pass


if __name__ == "__main__":
    run()
