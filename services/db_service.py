import logging
import os

import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)


def get_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASS"),
        dbname=os.getenv("DB_NAME"),
    )


def execute_query(sql: str) -> list[dict]:
    """
    Executa uma query SELECT no PostgreSQL e retorna os resultados como lista de dicionários.
    Raises QueryError com o SQL e a mensagem de erro original para permitir retry externo.
    """
    conn = None
    try:
        conn = get_connection()
        conn.set_session(readonly=True, autocommit=True)

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            log.info("Executando SQL:\n%s", sql)
            cur.execute(sql)
            rows = cur.fetchall()
            log.info("Query retornou %d linha(s).", len(rows))
            return [dict(row) for row in rows]

    except psycopg2.Error as e:
        # Log completo do erro + SQL para facilitar debug
        log.error("Erro ao executar SQL:\n%s\n\nErro psycopg2: %s", sql, e)
        raise QueryError(sql=sql, db_error=str(e)) from e
    finally:
        if conn:
            conn.close()


class QueryError(Exception):
    """Carrega o SQL que falhou e a mensagem de erro do banco para uso no retry."""

    def __init__(self, sql: str, db_error: str):
        self.sql = sql
        self.db_error = db_error
        super().__init__(f"Erro ao executar SQL: {db_error}")
