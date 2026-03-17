import json
import logging
import os
from datetime import datetime

from openai import OpenAI

log = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4o-mini"


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: formatação monetária brasileira
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_brl(value: float) -> str:
    """Converte float para R$ 1.234.567,89 (padrão pt-BR)."""
    s = f"{value:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"


def _fmt_int(value: int) -> str:
    """Converte int para 1.234 (separador de milhar pt-BR)."""
    return f"{value:,}".replace(",", ".")


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT A — Resolvedor de Intenção + Roteador
# Responsabilidade: recebe o histórico bruto e devolve UMA das três saídas:
#   1. RESPOSTA_DIRETA: <texto>  → pergunta conversacional
#   2. CLARIFICACAO: <texto>     → pergunta ambígua
#   3. Intent standalone         → consulta de dados (vai para o gerador SQL)
# ─────────────────────────────────────────────────────────────────────────────

def _build_resolver_prompt() -> str:
    hoje = datetime.now()
    return f"""Você é um roteador e resolvedor de intenção para um sistema de análise de dados.
Hoje é {hoje.strftime('%Y-%m-%d')} ({hoje.strftime('%A')}).

Analise a ÚLTIMA mensagem do usuário no histórico e retorne EXATAMENTE um dos três formatos:

━━━ CASO A — Pergunta conversacional sobre dados já exibidos ━━━
(ex: "qual mês foi esse?", "o que significa esse número?", "pode explicar?")
→ Retorne: RESPOSTA_DIRETA: <resposta direta baseada no histórico>

━━━ CASO B — Pergunta ambígua ou incompleta ━━━
Apenas quando falta informação que o usuário PRECISA fornecer (ex: "faturamento do mês de 2024" — qual mês?).
NÃO use este caso para expressões relativas como:
"este mês", "esse mês", "esse ano", "hoje", "mês passado", "últimos 7 dias",
"últimas 2 semanas", "últimos 3 meses" — todas são resolvíveis com CURRENT_DATE.
→ Retorne: CLARIFICACAO: <pergunta pedindo o dado faltante>

━━━ CASO C — Consulta de dados (requer SQL) ━━━
Retorne UMA ÚNICA FRASE standalone que descreva com precisão o que o usuário quer,
resolvendo TODOS os pronomes e referências temporais com base no histórico.
Regras:
- "desse período/mês/ano" → substitua pelo período EXATO citado no histórico
- "desse cliente/produto/vendedor" → substitua pela entidade EXATA citada no histórico
- A frase deve ser 100% auto-suficiente, sem pronomes relativos ou referências ao diálogo
- Inclua: métrica desejada + entidade (se houver) + período (se houver)

Exemplos:
  Histórico: perguntou faturamento de janeiro/2026. Atual: "e em fevereiro?"
  → Faturamento total de fevereiro de 2026

  Histórico: relatório de vendedores em março/2026. Atual: "e o ticket médio desse período?"
  → Ticket médio por vendedor de março de 2026

  Histórico: nenhum. Atual: "qual o meu faturamento esse mês?" ou "esse mês" ou "este mês"
  → Faturamento total de {hoje.strftime('%B de %Y')}

  Histórico: nenhum. Atual: "relatório deste mês"
  → Relatório de faturamento de {hoje.strftime('%B de %Y')} agrupado por produto

  Histórico: nenhum. Atual: "qual o faturamento dos últimos 15 dias?"
  → Faturamento total dos últimos 15 dias (de {hoje.strftime('%Y-%m-%d')} - 15 dias até hoje)

  Histórico: nenhum. Atual: "vendas das últimas 2 semanas"
  → Faturamento total das últimas 2 semanas até hoje

  Histórico: conversa sobre janeiro de 2026. Atual: "qual o melhor cliente que me comprou no mês de janeiro?"
  → Top clientes por faturamento em janeiro de 2026

  Histórico: conversa sobre janeiro de 2026 por cidade. Atual: "qual o melhor cliente de fortaleza nesse período?"
  → Melhor cliente por faturamento em Fortaleza em janeiro de 2026"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT B — Gerador de SQL puro
# Recebe APENAS a intent standalone resolvida. Nunca vê o histórico bruto.
# ─────────────────────────────────────────────────────────────────────────────

def _build_sql_prompt() -> str:
    hoje = datetime.now()
    return f"""Você é um especialista em SQL para PostgreSQL.
Você receberá uma instrução de consulta clara e auto-suficiente.
Sua única função é convertê-la em uma query SQL válida.

🔴 STRICT SCHEMA BINDING — CRÍTICO
════════════════════════════════════════
AS ÚNICAS TABELAS PERMITIDAS SÃO:
  • integralmix.agg_vendas_diarias       ← tabela fato agregada (preferencial)
  • integralmix."fVendas"                ← tabela fato bruta (fallback)
  • integralmix."INT_dim_Objetos"        ← dimensão de produtos (para JOINs)
  • integralmix."GR_dim_Clientes"        ← dimensão de clientes (para JOINs)
É ESTRITAMENTE PROIBIDO inventar ou referenciar qualquer outra tabela.
SEMPRE inclua o prefixo do schema integralmix. na query.
Se não for possível responder com essas quatro tabelas, retorne: PERGUNTA_INVALIDA

════════════════════════════════════════
CONTEXTO TEMPORAL
════════════════════════════════════════
Hoje é {hoje.strftime('%Y-%m-%d')}. Ano vigente: {hoje.strftime('%Y')}.

▶ DATAS RELATIVAS (palavras como "este mês", "hoje", "ano passado"):
  - "hoje"        → data_emissao = CURRENT_DATE
  - "este mês"    → DATE_TRUNC('month', data_emissao) = DATE_TRUNC('month', CURRENT_DATE)
  - "mês passado" → DATE_TRUNC('month', data_emissao) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  - "este ano"        → EXTRACT(YEAR FROM data_emissao) = EXTRACT(YEAR FROM CURRENT_DATE)
  - "ano passado"     → EXTRACT(YEAR FROM data_emissao) = EXTRACT(YEAR FROM CURRENT_DATE) - 1
  - "últimos N dias"  → data_emissao >= CURRENT_DATE - INTERVAL 'N days'
  - "últimas N semanas" → data_emissao >= CURRENT_DATE - INTERVAL 'N weeks'
  - "últimos N meses" → data_emissao >= CURRENT_DATE - INTERVAL 'N months'

▶ DATAS EXPLÍCITAS (ano ou mês específico já resolvido na instrução):
  É PROIBIDO usar CURRENT_DATE ou subtração matemática.
  - "janeiro de 2026" → data_emissao >= '2026-01-01' AND data_emissao < '2026-02-01'
  - "ano de 2025"     → EXTRACT(YEAR FROM data_emissao) = 2025

════════════════════════════════════════
FONTES DE DADOS
════════════════════════════════════════

▶ PREFERENCIAL — integralmix.agg_vendas_diarias
  Colunas:
    data_emissao (date), mes_emissao (date),
    "CodigoEmpresa", "Empresa", "EmpresaGerencial",
    "CodigoCliente", "Cliente",
    "CodigoObjeto", "Objeto", "CodigoObjetoMae", "ObjetoMae",
    "CodigoVendedor", "Vendedor", "Gerente", "Supervisor",
    "Estado", "Cidade", "CanalCliente", "Segmento", "AreaVenda", "AreaVendaMae",
    "FormadePagamento", "TipodeDocumento",
    faturamento, qtd_pedidos, qtd_clientes, qtd_produtos, volume_toneladas, ticket_medio

  Métricas pré-calculadas (use SUM para re-agregar):
    SUM(faturamento), SUM(volume_toneladas),
    COUNT(DISTINCT "CodigoCliente"), COUNT(DISTINCT "CodigoObjeto")

  ⚠ VOLUME DE PEDIDOS — NUNCA use SUM(qtd_pedidos) da view agregada para contar pedidos.
    A soma perde a distinção e duplica contagens. Para qtd de pedidos, sempre use:
    (SELECT COUNT(DISTINCT "Lancamento") FROM integralmix."fVendas" WHERE <mesmos filtros>)
    OU faça a query principal direto na integralmix."fVendas" com COUNT(DISTINCT "Lancamento").

▶ FALLBACK — integralmix."fVendas"
  Use apenas quando precisar de CEP, Latitude, Longitude, Prazo ou Lancamento individual.
  Métricas: SUM("Valor"), COUNT(DISTINCT "Lancamento"), SUM("Qtd")/1000.0

▶ DIMENSÃO DE PRODUTOS — integralmix."INT_dim_Objetos" (alias: dim)
  Chave: "CodigoObjeto". Colunas: "Objeto", "DivisaoObjeto", "LinhaObjeto",
  "DescricaoGerencial", "TipoObjeto", "Ativo", "Nivel02"
  Filtro obrigatório: dim."Nivel02" = 'Produtos p/ venda'
  JOIN: ... JOIN integralmix."INT_dim_Objetos" AS dim ON v."CodigoObjeto" = dim."CodigoObjeto"

▶ DIMENSÃO DE CLIENTES — integralmix."GR_dim_Clientes" (alias: cli)
  Chave: "CodigoCliente". Colunas: "RazaoSocial", "DivisaoCliente",
  "Estado", "Municipio", "TipoEstabelecimento", "Grupo_Empresa"
  JOIN: ... JOIN integralmix."GR_dim_Clientes" AS cli ON v."CodigoCliente" = cli."CodigoCliente"

════════════════════════════════════════
REGRAS DE SAÍDA
════════════════════════════════════════
1. Retorne APENAS a string SQL pura. Sem markdown, sem ```sql, sem explicações.
2. Aspas duplas em colunas com maiúsculas. Colunas minúsculas da view não precisam.
3. NUNCA gere INSERT, UPDATE, DELETE, DROP, CREATE ou qualquer DDL/DML.
4. Filtros de texto: SEMPRE use ILIKE com % (ex: WHERE "Cliente" ILIKE '%nome%').
5. Se a instrução não puder ser respondida com as tabelas disponíveis: PERGUNTA_INVALIDA
6. 🔴 REPORT MODE: Para perguntas amplas sem filtro de entidade específica, NUNCA retorne
   apenas um SUM() isolado. Use GROUP BY pela dimensão mais relevante ("Objeto", "Estado"
   ou "Vendedor") + SUM(faturamento) + SUM(qtd_pedidos), ORDER BY DESC LIMIT 20.

7. 🔴 LIMIT OBRIGATÓRIO: Sempre que a pergunta envolver "Top", "Ranking", "Maiores",
   "Melhores", ou for um relatório/overview genérico, inclua LIMIT 20 no final da query.

8. 🔴 TOTAL REAL COM WINDOW FUNCTION (CRÍTICO PARA CONSISTÊNCIA):
   SEMPRE que usar GROUP BY, inclua obrigatoriamente as colunas de total global via
   window function. Isso garante que o total correto seja preservado mesmo com LIMIT.
   Modelo obrigatório para queries com GROUP BY:
     SUM(SUM("Valor")) OVER () AS total_geral_faturamento
   Essa coluna terá o mesmo valor em todas as linhas e representa o faturamento REAL
   de todo o período/filtro, antes do LIMIT ser aplicado pelo banco.
   ⚠ NÃO use COUNT(DISTINCT col) OVER () — é inválido no PostgreSQL.

════════════════════════════════════════
EXEMPLOS
════════════════════════════════════════

-- Instrução: "Faturamento total de fevereiro de 2026"
SELECT SUM(faturamento)                          AS faturamento_total,
       COUNT(DISTINCT "Lancamento")              AS qtd_pedidos
FROM integralmix."fVendas"
WHERE "DataEmissao" >= '2026-02-01' AND "DataEmissao" < '2026-03-01';

---

-- Instrução: "Relatório de faturamento de março de 2026 agrupado por produto"
SELECT "Objeto",
       SUM("Valor")                              AS faturamento_total,
       COUNT(DISTINCT "Lancamento")              AS qtd_pedidos,
       SUM(SUM("Valor")) OVER ()                  AS total_geral_faturamento
FROM integralmix."fVendas"
WHERE "DataEmissao" >= '2026-03-01' AND "DataEmissao" < '2026-04-01'
GROUP BY "Objeto"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Instrução: "Faturamento por estado dos últimos 15 dias"
SELECT "Estado",
       SUM("Valor")                              AS faturamento_total,
       COUNT(DISTINCT "Lancamento")              AS qtd_pedidos,
       SUM(SUM("Valor")) OVER ()                  AS total_geral_faturamento
FROM integralmix."fVendas"
WHERE "DataEmissao" >= CURRENT_DATE - INTERVAL '15 days'
GROUP BY "Estado"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Instrução: "Faturamento do cliente Farelo Verde no ano de 2025"
SELECT SUM("Valor")                AS faturamento_total,
       COUNT(DISTINCT "Lancamento") AS qtd_pedidos
FROM integralmix."fVendas"
WHERE "Cliente" ILIKE '%farelo verde%'
  AND EXTRACT(YEAR FROM "DataEmissao") = 2025;

---

-- Instrução: "Ticket médio por vendedor deste mês"
SELECT "Vendedor",
       SUM("Valor") / NULLIF(COUNT(DISTINCT "Lancamento"), 0)  AS ticket_medio,
       SUM("Valor")                                             AS faturamento_total,
       COUNT(DISTINCT "Lancamento")                             AS qtd_pedidos,
       SUM(SUM("Valor"))       OVER ()                         AS total_geral_faturamento,
       COUNT(DISTINCT "Lancamento") OVER ()                    AS total_geral_pedidos
FROM integralmix."fVendas"
WHERE DATE_TRUNC('month', "DataEmissao") = DATE_TRUNC('month', CURRENT_DATE)
GROUP BY "Vendedor"
ORDER BY ticket_medio DESC
LIMIT 20;

---

-- Instrução: "Top clientes por faturamento em janeiro de 2026"
SELECT "Cliente",
       SUM("Valor")                                    AS faturamento_total,
       COUNT(DISTINCT "Lancamento")                    AS qtd_pedidos,
       SUM(SUM("Valor"))       OVER ()                 AS total_geral_faturamento,
       COUNT(DISTINCT "Lancamento") OVER ()            AS total_geral_pedidos
FROM integralmix."fVendas"
WHERE "DataEmissao" >= '2026-01-01' AND "DataEmissao" < '2026-02-01'
GROUP BY "Cliente"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Instrução: "Melhor cliente em Fortaleza em janeiro de 2026"
SELECT "Cliente",
       SUM("Valor")                                    AS faturamento_total,
       COUNT(DISTINCT "Lancamento")                    AS qtd_pedidos,
       SUM(SUM("Valor"))       OVER ()                 AS total_geral_faturamento,
       COUNT(DISTINCT "Lancamento") OVER ()            AS total_geral_pedidos
FROM integralmix."fVendas"
WHERE "Cidade" ILIKE '%fortaleza%'
  AND "DataEmissao" >= '2026-01-01' AND "DataEmissao" < '2026-02-01'
GROUP BY "Cliente"
ORDER BY faturamento_total DESC
LIMIT 20;
"""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT C — Extrator de Card Data (JSON estruturado)
# Responsabilidade: extrair números reais dos resultados. Nunca inventar.
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_CARD_EXTRACTOR = """Você é um extrator de dados estruturados.
Receberá o resultado bruto de uma consulta SQL (lista de dicionários Python) e a intenção da consulta.
Extraia os dados e retorne APENAS um JSON válido neste schema exato:

{
  "faturamento_total": <float ou null>,
  "qtd_pedidos": <int ou null>,
  "ticket_medio": <float ou null>,
  "top_drivers": [
    {"nome": "<string>", "faturamento": <float>, "qtd_pedidos": <int ou null>}
  ]
}

REGRAS CRÍTICAS — TOLERÂNCIA ZERO PARA ALUCINAÇÃO:
- Se um campo NÃO está nos dados recebidos → use null. NUNCA invente valores.

- faturamento_total:
    PRIORIDADE 1: Se existir a coluna "total_geral_faturamento" nos dados, use esse valor
    (é o mesmo em todas as linhas e representa o total real antes do LIMIT).
    PRIORIDADE 2: Caso contrário, some todos os valores de "faturamento_total" ou "faturamento".

- qtd_pedidos:
    PRIORIDADE 1: Se existir "total_geral_pedidos", use esse valor.
    PRIORIDADE 2: Caso contrário, some os valores de pedidos das linhas, ou null se não existir.

- ticket_medio: calcule faturamento_total / qtd_pedidos APENAS se ambos existirem; caso contrário null.

- top_drivers: se os dados têm agrupamento por entidade (produto, vendedor, estado, cidade, cliente, etc.),
  inclua os 3 primeiros itens ordenados do maior para o menor faturamento.
  O campo "nome" deve ser o VALOR da coluna dimensional — ou seja, o conteúdo real da célula
  (ex: "Fortaleza", "João da Silva", "Produto ABC"), NUNCA o nome da coluna (ex: nunca use
  "Cidade", "Cliente", "Objeto" como nome — esses são cabeçalhos, não valores).
  Se não há agrupamento (apenas 1 linha com total), use [].

- Retorne APENAS o JSON, sem markdown, sem explicações."""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT D — Correção de SQL (self-healing)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_FIX = """Você é um especialista em SQL para PostgreSQL.
Receberá uma query SQL que falhou e a mensagem de erro do banco.
Corrija o SQL e retorne APENAS a string SQL corrigida, sem markdown, sem explicações."""


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: Python monta o card — zero chance de alucinação
# ─────────────────────────────────────────────────────────────────────────────

def render_card(card_data: dict) -> str:
    """Monta o card Markdown a partir do JSON estruturado.
    Cada campo é exibido apenas se não for None — Python decide, não o LLM."""
    lines = ["### 🎯 Resumo Executivo"]

    fat = card_data.get("faturamento_total")
    ped = card_data.get("qtd_pedidos")
    tkt = card_data.get("ticket_medio")

    if fat is not None:
        lines.append(f"- 💰 Faturamento: {_fmt_brl(fat)}")
    if ped is not None:
        lines.append(f"- 📦 Volume: {_fmt_int(int(ped))} pedidos")
    if tkt is not None:
        lines.append(f"- 🎟️ Ticket Médio: {_fmt_brl(tkt)}")

    drivers = card_data.get("top_drivers") or []
    if drivers:
        medals = ["🥇 1º", "🥈 2º", "🥉 3º"]
        lines.append("\n### 🏆 Top 3 Drivers")
        for i, d in enumerate(drivers[:3]):
            medal = medals[i] if i < len(medals) else f"{i + 1}º"
            lines.append(f"{medal} - {d['nome']} - {_fmt_brl(d['faturamento'])}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def resolve_intent(history: list[dict]) -> str:
    """Passo 1: Roteia e resolve o intent do usuário a partir do histórico.
    Retorna RESPOSTA_DIRETA:, CLARIFICACAO: ou uma frase standalone limpa."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _build_resolver_prompt()},
            *history,
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def generate_sql(standalone_intent: str) -> str:
    """Passo 2: Gera SQL a partir de uma instrução standalone já resolvida.
    Não recebe histórico — sem contaminação de contexto."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _build_sql_prompt()},
            {"role": "user", "content": standalone_intent},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def fix_sql(broken_sql: str, db_error: str) -> str:
    """Self-healing: envia SQL com erro + mensagem do banco para correção (1 retry)."""
    log.warning("Tentando auto-correção de SQL via IA...")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT_FIX},
            {"role": "user", "content": f"SQL com erro:\n{broken_sql}\n\nErro:\n{db_error}"},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


_MAX_ROWS_TO_LLM = 20  # trava de payload: o LLM nunca vê mais que isso


def extract_card_data(standalone_intent: str, rows: list) -> dict:
    """Passo 3a: LLM extrai JSON estruturado dos resultados brutos.
    Trunca o payload antes de enviar para evitar estouro de tokens (429).
    Usa json_object mode — garante JSON válido na saída."""
    safe_rows = rows[:_MAX_ROWS_TO_LLM]
    if len(rows) > _MAX_ROWS_TO_LLM:
        log.info("Payload truncado: %d → %d linhas antes de enviar ao LLM.", len(rows), _MAX_ROWS_TO_LLM)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT_CARD_EXTRACTOR},
            {"role": "user", "content": f"Intenção: {standalone_intent}\n\nDados: {safe_rows}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.error("Falha ao parsear JSON do extrator: %s", raw)
        return {}
