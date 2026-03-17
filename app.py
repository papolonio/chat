import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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

from services.ai_service import resolve_intent, generate_sql, fix_sql, extract_card_data, render_card, summarize_for_history
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
SESSION_TTL = 3600       # 1 hora
MAX_HISTORY_MSGS = 10    # 5 turnos de conversa (user + assistant)


def _get_history(session_id: str) -> list[dict]:
    raw = redis_client.get(f"session:{session_id}")
    return json.loads(raw) if raw else []


# ---------------------------------------------------------------------------
# Relatório completo
# ---------------------------------------------------------------------------

_RELATORIO_DIMENSOES = [
    ("produto",   "Top produtos por faturamento em {periodo}"),
    ("cliente",   "Top clientes por faturamento em {periodo}"),
    ("vendedor",  "Top vendedores por faturamento em {periodo}"),
    ("estado",    "Faturamento por estado em {periodo}"),
]

_CABECALHOS = {
    "produto":  "## 📦 Produtos",
    "cliente":  "## 👥 Clientes",
    "vendedor": "## 🤝 Vendedores",
    "estado":   "## 🗺️ Estados / Regiões",
}


def _executar_dimensao(label: str, intent: str) -> tuple[str, str]:
    """Executa o pipeline completo (SQL → banco → card) para uma dimensão.
    Retorna (label, card_markdown). Em caso de erro retorna card com aviso."""
    cabecalho = _CABECALHOS.get(label, f"## {label.title()}")
    try:
        sql = generate_sql(intent)
        if sql.strip().upper() == "PERGUNTA_INVALIDA":
            raise ValueError("SQL gerou PERGUNTA_INVALIDA")
        try:
            rows = execute_query(sql)
        except QueryError as e:
            sql_corrigido = fix_sql(broken_sql=e.sql, db_error=e.db_error)
            rows = execute_query(sql_corrigido)
        if not rows:
            return label, f"{cabecalho}\n\n_Sem dados disponíveis para {label} neste período._"
        card_data = extract_card_data(intent, rows)
        return label, f"{cabecalho}\n\n{render_card(card_data)}"
    except Exception as exc:
        log.error("Relatório completo — falha em '%s': %s", label, exc)
        return label, f"{cabecalho}\n\n⚠️ Não foi possível carregar dados de **{label}** para este período."


def _save_history(session_id: str, history: list[dict]) -> None:
    # Trunca antes de persistir — garante que o Redis nunca cresce além do limite
    trimmed = history[-MAX_HISTORY_MSGS:]
    redis_client.setex(f"session:{session_id}", SESSION_TTL, json.dumps(trimmed))


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
    #    Trunca o histórico antes de enviar à IA para evitar estouro de tokens
    trimmed = history[-MAX_HISTORY_MSGS:] if len(history) > MAX_HISTORY_MSGS else history
    if len(history) > MAX_HISTORY_MSGS:
        log.info("Histórico truncado: %d → %d msgs para o resolvedor.", len(history), MAX_HISTORY_MSGS)
    resolved = resolve_intent(trimmed)
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

    if resolved.upper().startswith("RELATORIO_COMPLETO:"):
        periodo = resolved[len("RELATORIO_COMPLETO:"):].strip()
        log.info("Relatório completo solicitado para: %s", periodo)

        intents = [(label, tmpl.format(periodo=periodo)) for label, tmpl in _RELATORIO_DIMENSOES]
        resultados: dict[str, str] = {}

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(_executar_dimensao, label, intent): label
                       for label, intent in intents}
            for future in as_completed(futures):
                label, card = future.result()
                resultados[label] = card

        # Mantém a ordem original das dimensões
        cards_ordenados = [resultados[label] for label, _ in _RELATORIO_DIMENSOES]
        answer = "\n\n---\n\n".join(cards_ordenados)

        history_summary = f"Relatório completo de {periodo} (produto, cliente, vendedor, estado)."
        history.append({"role": "assistant", "content": history_summary})
        _save_history(session_id, history)
        return jsonify({"answer": answer})

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

    # Gera chart_data se houver série temporal (para gráfico no frontend)
    serie = card_data.get("serie_temporal") or []
    chart_data = None
    if len(serie) >= 2:
        chart_data = {
            "labels": [s["periodo"] for s in serie],
            "values": [s.get("faturamento", 0) for s in serie],
        }
        log.info("chart_data gerado com %d pontos.", len(serie))

    # Salva no histórico um resumo compacto (não o card Markdown completo)
    # para não poluir o contexto enviado ao resolver nos turnos seguintes.
    history_content = summarize_for_history(card_data, resolved)
    history.append({"role": "assistant", "content": history_content})
    _save_history(session_id, history)

    return jsonify({"answer": answer, "chart_data": chart_data})


if __name__ == "__main__":
    app.run(debug=True)
