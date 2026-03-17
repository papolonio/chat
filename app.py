import json
import logging
import os

import redis
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

from services.ai_service import resolve_intent, generate_sql, fix_sql, extract_card_data, render_card
from services.db_service import execute_query, QueryError

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Redis
# ---------------------------------------------------------------------------
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=0,
    decode_responses=True,
)
SESSION_TTL = 3600  # 1 hora


def _get_history(session_id: str) -> list[dict]:
    raw = redis_client.get(f"session:{session_id}")
    return json.loads(raw) if raw else []


def _save_history(session_id: str, history: list[dict]) -> None:
    redis_client.setex(f"session:{session_id}", SESSION_TTL, json.dumps(history))


# ---------------------------------------------------------------------------
# Rotas
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    message = (data.get("message") or "").strip()
    session_id = (data.get("session_id") or "").strip()

    if not message:
        return jsonify({"error": "Mensagem vazia."}), 400
    if not session_id:
        return jsonify({"error": "session_id ausente."}), 400

    # 1. Busca histórico no Redis e anexa nova mensagem
    history = _get_history(session_id)
    history.append({"role": "user", "content": message})

    # 2. RESOLUÇÃO DE INTENÇÃO — roteamento + isolamento de contexto
    #    O resolvedor lê o histórico bruto e devolve:
    #      - RESPOSTA_DIRETA: <texto>  → pergunta conversacional
    #      - CLARIFICACAO: <texto>     → pergunta ambígua
    #      - <frase standalone>        → intent limpo para geração de SQL
    resolved = resolve_intent(history)
    log.info("Intent resolvida: %s", resolved[:120])

    if resolved.upper().startswith("RESPOSTA_DIRETA:"):
        answer = resolved[len("RESPOSTA_DIRETA:"):].strip()
        history.append({"role": "assistant", "content": answer})
        _save_history(session_id, history)
        return jsonify({"answer": answer})

    if resolved.upper().startswith("CLARIFICACAO:"):
        answer = resolved[len("CLARIFICACAO:"):].strip()
        history.append({"role": "assistant", "content": answer})
        _save_history(session_id, history)
        return jsonify({"answer": answer, "clarification": True})

    # 3. GERAÇÃO DE SQL — recebe apenas o intent standalone (sem histórico bruto)
    #    Elimina context bleeding: o gerador nunca vê perguntas anteriores.
    sql = generate_sql(resolved)
    log.info("SQL gerado:\n%s", sql)

    if sql.strip().upper() == "PERGUNTA_INVALIDA":
        answer = (
            "Não consegui entender sua pergunta ou ela está fora do escopo dos dados disponíveis. "
            "Tente perguntar sobre faturamento, clientes, vendedores ou produtos."
        )
        history.append({"role": "assistant", "content": answer})
        _save_history(session_id, history)
        return jsonify({"answer": answer})

    # 4. EXECUÇÃO NO BANCO — com 1 retry automático de self-healing
    rows = None
    try:
        rows = execute_query(sql)
    except QueryError as e:
        log.warning("Falha na execução. Iniciando self-healing...")
        try:
            sql_corrigido = fix_sql(broken_sql=e.sql, db_error=e.db_error)
            log.info("SQL corrigido:\n%s", sql_corrigido)
            rows = execute_query(sql_corrigido)
        except QueryError as e2:
            log.error("Self-healing falhou. SQL: %s | Erro: %s", e2.sql, e2.db_error)
            return jsonify({"error": "Não consegui executar a consulta após tentativa de correção automática."}), 500

    # 5. SEM DADOS
    if not rows:
        answer = "A consulta foi executada mas não retornou dados para o período ou filtros informados."
        history.append({"role": "assistant", "content": answer})
        _save_history(session_id, history)
        return jsonify({"answer": answer})

    # 6. EXTRAÇÃO ESTRUTURADA + RENDERIZAÇÃO PYTHON
    #    LLM extrai apenas JSON com os números que existem nos dados.
    #    Python monta o card — zero possibilidade de alucinação de valores.
    card_data = extract_card_data(resolved, rows)
    answer = render_card(card_data)

    # Fallback: se extração falhou completamente, usa str dos dados brutos
    if not answer.strip() or answer == "### 🎯 Resumo Executivo":
        log.warning("Card vazio após extração. Usando fallback de dados brutos.")
        answer = f"Dados retornados:\n{rows}"

    history.append({"role": "assistant", "content": answer})
    _save_history(session_id, history)

    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
