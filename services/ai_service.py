import json
import logging
import os
from datetime import datetime

from openai import OpenAI

log = logging.getLogger(__name__)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Melhoria #3: modelo maior apenas no Prompt A (roteamento mais crítico)
MODEL_ROUTER = "gpt-4o"
MODEL_SQL    = "gpt-4o-mini"
MODEL_CARD   = "gpt-4o-mini"
MODEL_FIX    = "gpt-4o-mini"


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

# Parte estática cacheável (não muda entre requests)
_RESOLVER_STATIC = """Você é um roteador e resolvedor de intenção para um sistema de análise de dados.

Analise a ÚLTIMA mensagem do usuário no histórico e retorne EXATAMENTE um dos três formatos:

━━━ CASO A — Pergunta conversacional sobre dados já exibidos ━━━
(ex: "qual mês foi esse?", "o que significa esse número?", "pode explicar?")
→ Retorne: RESPOSTA_DIRETA: <resposta direta baseada no histórico>

━━━ CASO B — Pergunta ambígua ou incompleta ━━━
Apenas quando falta informação que o usuário PRECISA fornecer (ex: "faturamento do mês de 2024" — qual mês?).
NÃO use este caso para:
- Expressões relativas: "este mês", "esse mês", "esse ano", "hoje", "mês passado",
  "últimos 7 dias", "últimas 2 semanas", "últimos 3 meses" — resolvíveis com CURRENT_DATE.
- Mês + ano explícito: "janeiro de 2026", "fevereiro de 2026", "março de 2025" etc.
  → Mês + ano = período completo do mês inteiro. NUNCA pergunte o dia específico.
- Ano explícito: "em 2025", "no ano de 2024" → período do ano inteiro.
→ Retorne: CLARIFICACAO: <pergunta pedindo o dado faltante>

━━━ CASO C — Consulta de dados (requer SQL) ━━━
Retorne UMA ÚNICA FRASE standalone que descreva com precisão o que o usuário quer,
resolvendo TODOS os pronomes e referências temporais com base no histórico.
Regras:
- "desse período/mês/ano" → substitua pelo período EXATO citado no histórico
- "desse cliente/produto/vendedor" → substitua pela entidade EXATA citada no histórico
- "agora", "quero ver", "me mostre", "e por" → carregue a DIMENSÃO ATIVA da query anterior
  (ex: se o histórico mostrou produtos, "quero ver por ticket médio" = ticket médio POR PRODUTO)
- A frase deve ser 100% auto-suficiente, sem pronomes relativos ou referências ao diálogo
- Inclua: métrica desejada + DIMENSÃO (produto/cliente/vendedor/estado) + período

Exemplos:
  Histórico: perguntou faturamento de janeiro/2026. Atual: "e em fevereiro?"
  → Faturamento total de fevereiro de 2026

  Histórico: relatório de vendedores em março/2026. Atual: "e o ticket médio desse período?"
  → Ticket médio por vendedor de março de 2026

  Histórico: nenhum. Atual: "qual o meu faturamento esse mês?" ou "esse mês" ou "este mês"
  → Faturamento total de {mes_atual}

  Histórico: nenhum. Atual: "relatório deste mês"
  → Relatório de faturamento de {mes_atual} agrupado por produto

  Histórico: nenhum. Atual: "qual o faturamento dos últimos 15 dias?"
  → Faturamento total dos últimos 15 dias (de {data_atual} - 15 dias até hoje)

  Histórico: nenhum. Atual: "vendas das últimas 2 semanas"
  → Faturamento total das últimas 2 semanas até hoje

  Histórico: conversa sobre janeiro de 2026. Atual: "qual o melhor cliente que me comprou no mês de janeiro?"
  → Top clientes por faturamento em janeiro de 2026

  Histórico: conversa sobre janeiro de 2026 por cidade. Atual: "qual o melhor cliente de fortaleza nesse período?"
  → Melhor cliente por faturamento em Fortaleza em janeiro de 2026

  Histórico: faturamento de março/2026 = R$30M. Atual: "e em comparação ao mês anterior?" ou "e versus fevereiro?"
  → Comparação de faturamento entre fevereiro de 2026 e março de 2026

  Histórico: faturamento dos últimos 15 dias = R$30M. Atual: "e em comparação ao período anterior?"
  → Comparação de faturamento: últimos 15 dias vs 15 dias anteriores

  Histórico: nenhum. Atual: "como foi a evolução do faturamento mês a mês este ano?" ou "crescimento mensal"
  → Evolução mensal do faturamento no ano de 2026

  Histórico: faturamento dos últimos 15 dias = R$30M. Atual: "e quais foram os produtos vendidos nesse mesmo período?" ou "e os produtos nesse período?"
  → Top produtos por faturamento dos últimos 15 dias

  Histórico: faturamento de janeiro de 2026. Atual: "e os produtos vendidos em janeiro de 2026?" ou "e os produtos?"
  → Top produtos por faturamento em janeiro de 2026

  Histórico: relatório de clientes de março/2026. Atual: "e os vendedores nesse mesmo período?"
  → Top vendedores por faturamento em março de 2026

  Histórico: relatório de PRODUTOS de janeiro/2026. Atual: "quero ver por ticket médio agora" ou "e o ticket médio?"
  → Ticket médio por produto em janeiro de 2026

  Histórico: relatório de PRODUTOS dos últimos 15 dias. Atual: "e o volume de pedidos?" ou "quero ver por pedidos agora"
  → Volume de pedidos por produto dos últimos 15 dias

  Histórico: relatório de VENDEDORES de fevereiro/2026. Atual: "quero ver por ticket médio"
  → Ticket médio por vendedor em fevereiro de 2026

  Histórico: relatório de ESTADOS (regiões) de março/2026. Atual: "e por ticket médio?"
  → Ticket médio por estado em março de 2026

  Histórico: relatório de CLIENTES de janeiro/2026. Atual: "e por região?" ou "e por estado?"
  → Faturamento por estado em janeiro de 2026

  Histórico: relatório de CLIENTES de janeiro/2026. Atual: "e por produto?" ou "e os produtos?"
  → Top produtos por faturamento em janeiro de 2026

  Histórico: relatório de CLIENTES de janeiro/2026. Atual: "e por vendedor?"
  → Top vendedores por faturamento em janeiro de 2026

  Histórico: relatório de PRODUTOS de janeiro/2026. Atual: "e por cliente?" ou "e os clientes?"
  → Top clientes por faturamento em janeiro de 2026

  Histórico: relatório de PRODUTOS de janeiro/2026. Atual: "e por região?" ou "e por estado?"
  → Faturamento por estado em janeiro de 2026"""


def _build_resolver_prompt() -> str:
    """Monta o prompt do roteador injetando apenas a parte dinâmica (data atual)."""
    hoje = datetime.now()
    return _RESOLVER_STATIC.format(
        mes_atual=hoje.strftime("%B de %Y"),
        data_atual=hoje.strftime("%Y-%m-%d"),
    ) + f"\n\nHoje é {hoje.strftime('%Y-%m-%d')} ({hoje.strftime('%A')})."


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT B — Gerador de SQL puro
# Melhoria #1: regras absolutas e críticas movidas para o TOPO do prompt.
# Melhoria #7: glossário de negócio separado das regras sintáticas de SQL.
# Recebe APENAS a intent standalone resolvida. Nunca vê o histórico bruto.
# ─────────────────────────────────────────────────────────────────────────────

# Parte estática cacheável
_SQL_STATIC = """Você é um especialista em SQL para PostgreSQL.
Você receberá uma instrução de consulta clara e auto-suficiente.
Sua única função é convertê-la em uma query SQL válida.

════════════════════════════════════════
🔴 REGRAS ABSOLUTAS — LEIA PRIMEIRO
════════════════════════════════════════
1. Retorne APENAS a string SQL pura. Sem markdown, sem ```sql, sem explicações.
2. NUNCA gere INSERT, UPDATE, DELETE, DROP, CREATE ou qualquer DDL/DML.
3. Aspas duplas em colunas com maiúsculas. Colunas minúsculas da view não precisam.
4. Filtros de texto: SEMPRE use ILIKE com % (ex: WHERE "Cliente" ILIKE '%nome%').
5. SEMPRE que usar GROUP BY, inclua SUM(SUM(valor)) OVER () AS total_geral_faturamento.
6. SEMPRE que a query for ranking/top/maiores/relatório genérico, inclua LIMIT 20.
7. Se não for possível responder com as quatro tabelas permitidas → retorne: PERGUNTA_INVALIDA

════════════════════════════════════════
🔴 STRICT SCHEMA BINDING — TABELAS PERMITIDAS
════════════════════════════════════════
AS ÚNICAS TABELAS PERMITIDAS SÃO:
  • integralmix.agg_vendas_diarias       ← tabela fato agregada (preferencial)
  • integralmix."fVendas"                ← tabela fato bruta (fallback)
  • integralmix."INT_dim_Objetos"        ← dimensão de produtos (para JOINs)
  • integralmix."GR_dim_Clientes"        ← dimensão de clientes (para JOINs)
É ESTRITAMENTE PROIBIDO inventar ou referenciar qualquer outra tabela.
SEMPRE inclua o prefixo do schema integralmix. na query.

════════════════════════════════════════
SELEÇÃO DE TABELA — REGRA CRÍTICA
════════════════════════════════════════
▶ USE integralmix.agg_vendas_diarias QUANDO:
  - A consulta envolve apenas FATURAMENTO (SUM) e/ou agrupamentos por dimensão.
  - É muito mais rápida (dados pré-agregados). Prefira sempre que possível.
  - Coluna de valor:  faturamento  (minúsculo, SEM aspas duplas)
  - Coluna de data:   data_emissao (minúsculo, SEM aspas duplas)
  - ⚠ Não existe coluna "Valor" nem "DataEmissao" nesta tabela.

▶ USE integralmix."fVendas" QUANDO:
  - A consulta precisa de COUNT(DISTINCT "Lancamento") — contagem exata de pedidos.
  - A consulta precisa de Ticket Médio (que depende de pedidos exatos).
  - Coluna de valor:  "Valor"       (com aspas, V maiúsculo)
  - Coluna de data:   "DataEmissao" (com aspas, D e E maiúsculos)
  - ⚠ Não existe coluna faturamento (sem aspas) nesta tabela.

════════════════════════════════════════
CONTEXTO TEMPORAL
════════════════════════════════════════
▶ DATAS RELATIVAS:
  - "hoje"            → data_emissao = CURRENT_DATE
  - "este mês"        → DATE_TRUNC('month', data_emissao) = DATE_TRUNC('month', CURRENT_DATE)
  - "mês passado"     → DATE_TRUNC('month', data_emissao) = DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month')
  - "este ano"        → EXTRACT(YEAR FROM data_emissao) = EXTRACT(YEAR FROM CURRENT_DATE)
  - "ano passado"     → EXTRACT(YEAR FROM data_emissao) = EXTRACT(YEAR FROM CURRENT_DATE) - 1
  - "últimos N dias"  → data_emissao >= CURRENT_DATE - INTERVAL 'N days'
  - "últimas N semanas" → data_emissao >= CURRENT_DATE - INTERVAL 'N weeks'
  - "últimos N meses" → data_emissao >= CURRENT_DATE - INTERVAL 'N months'

▶ DATAS EXPLÍCITAS (período já resolvido na instrução):
  É PROIBIDO usar CURRENT_DATE ou subtração matemática.
  - "janeiro de 2026" → data_emissao >= '2026-01-01' AND data_emissao < '2026-02-01'
  - "ano de 2025"     → EXTRACT(YEAR FROM data_emissao) = 2025

════════════════════════════════════════
FONTES DE DADOS
════════════════════════════════════════
▶ integralmix.agg_vendas_diarias — PREFERENCIAL
  Colunas dimensionais (use com aspas):
    "CodigoEmpresa", "Empresa", "EmpresaGerencial",
    "CodigoCliente", "Cliente",
    "CodigoObjeto", "Objeto", "CodigoObjetoMae", "ObjetoMae",
    "CodigoVendedor", "Vendedor", "Gerente", "Supervisor",
    "Estado", "Cidade", "CanalCliente", "Segmento", "AreaVenda", "AreaVendaMae",
    "FormadePagamento", "TipodeDocumento"
  Colunas de data/métrica (sem aspas, minúsculas):
    data_emissao, mes_emissao, faturamento, volume_toneladas
  ⚠ qtd_pedidos nesta tabela NÃO deve ser somada entre grupos — duplica contagens.

▶ integralmix."fVendas" — PARA PEDIDOS E TICKET MÉDIO
  Coluna de valor: "Valor" (com aspas, V maiúsculo). Coluna de data: "DataEmissao".
  Métricas: SUM("Valor"), COUNT(DISTINCT "Lancamento"), SUM("Qtd")/1000.0

▶ integralmix."INT_dim_Objetos" (alias: dim)
  Chave: "CodigoObjeto"
  Filtro obrigatório: dim."Nivel02" = 'Produtos p/ venda'
  JOIN: ... JOIN integralmix."INT_dim_Objetos" AS dim ON v."CodigoObjeto" = dim."CodigoObjeto"

▶ integralmix."GR_dim_Clientes" (alias: cli)
  Chave: "CodigoCliente"
  JOIN: ... JOIN integralmix."GR_dim_Clientes" AS cli ON v."CodigoCliente" = cli."CodigoCliente"

════════════════════════════════════════
INTERPRETAÇÃO DE TERMOS DO USUÁRIO
════════════════════════════════════════
▶ Aliases de dimensão (palavra do usuário → coluna real):
  "região" / "regiões" / "regional" → "Estado"
  "produto" / "produtos" / "item"   → "Objeto"
  "vendedor" / "representante"      → "Vendedor"
  "cliente" / "clientes"            → "Cliente"
  "cidade" / "município"            → "Cidade"

▶ Dimensões em agg_vendas_diarias (sem JOIN):
  "canal" / "canal de venda"        → "CanalCliente"
  "segmento"                        → "Segmento"
  "área de venda"                   → "AreaVenda"
  "área mãe" / "área gerencial"     → "AreaVendaMae"
  "forma de pagamento"              → "FormadePagamento"
  "tipo de documento"               → "TipodeDocumento"
  "empresa" / "filial"              → "Empresa"
  "empresa gerencial"               → "EmpresaGerencial"
  "gerente"                         → "Gerente"
  "supervisor"                      → "Supervisor"
  "volume" / "toneladas"            → volume_toneladas
  "produto pai" / "família"         → "ObjetoMae"

▶ Dimensões via JOIN com INT_dim_Objetos:
  "linha do produto"                → dim."LinhaObjeto"
  "divisão do produto"              → dim."DivisaoObjeto"
  "descrição gerencial"             → dim."DescricaoGerencial"
  "tipo de produto"                 → dim."TipoObjeto"
  "produto ativo"                   → dim."Ativo" = true

▶ Dimensões via JOIN com GR_dim_Clientes:
  "razão social"                    → cli."RazaoSocial"
  "divisão do cliente"              → cli."DivisaoCliente"
  "tipo de estabelecimento"         → cli."TipoEstabelecimento"
  "grupo empresarial"               → cli."Grupo_Empresa"
  "município do cliente"            → cli."Municipio"

▶ Métricas:
  "faturamento" / "receita" / "vendas"         → SUM(faturamento) ou SUM("Valor")
  "ticket médio"                               → SUM("Valor") / COUNT(DISTINCT "Lancamento")
  "volume de pedidos" / "pedidos"              → COUNT(DISTINCT "Lancamento") via fVendas
  "clientes atendidos"                         → COUNT(DISTINCT "CodigoCliente")
  "mix de produtos"                            → COUNT(DISTINCT "CodigoObjeto")

▶ Intenções que NUNCA devem retornar PERGUNTA_INVALIDA:
  "produtos vendidos" / "top produtos"   → GROUP BY "Objeto"
  "melhores clientes" / "top clientes"   → GROUP BY "Cliente"
  "vendedores" / "top vendedores"        → GROUP BY "Vendedor"
  "por região" / "por estado"            → GROUP BY "Estado"
  "por cidade"                           → GROUP BY "Cidade"

════════════════════════════════════════
REGRAS SQL — CONSTRUÇÃO DA QUERY
════════════════════════════════════════
▶ TOTAL REAL COM WINDOW FUNCTION (CRÍTICO):
  SEMPRE que usar GROUP BY, inclua:
    SUM(SUM(faturamento)) OVER () AS total_geral_faturamento   ← para agg_vendas_diarias
    SUM(SUM("Valor"))     OVER () AS total_geral_faturamento   ← para fVendas
  Essa coluna tem o mesmo valor em todas as linhas e representa o total REAL antes do LIMIT.
  ⚠ NÃO use COUNT(DISTINCT col) OVER () — é inválido no PostgreSQL.

▶ REPORT MODE:
  Para perguntas amplas sem filtro de entidade específica, NUNCA retorne apenas um SUM() isolado.
  Use GROUP BY pela dimensão mais relevante + SUM + COUNT(DISTINCT "Lancamento") + ORDER BY + LIMIT 20.

▶ LIMIT OBRIGATÓRIO:
  Sempre que a pergunta envolver "Top", "Ranking", "Maiores", "Melhores" ou for um relatório
  genérico, inclua LIMIT 20 no final.

════════════════════════════════════════
EXEMPLOS
════════════════════════════════════════

-- Top produtos de janeiro de 2026
SELECT "Objeto",
       SUM(faturamento)                          AS faturamento_total,
       SUM(SUM(faturamento)) OVER ()             AS total_geral_faturamento
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= '2026-01-01' AND data_emissao < '2026-02-01'
GROUP BY "Objeto"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Faturamento total de março de 2026
SELECT SUM(faturamento) AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= '2026-03-01' AND data_emissao < '2026-04-01';

---

-- Faturamento por estado dos últimos 15 dias
SELECT "Estado",
       SUM(faturamento)                          AS faturamento_total,
       SUM(SUM(faturamento)) OVER ()             AS total_geral_faturamento
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= CURRENT_DATE - INTERVAL '15 days'
GROUP BY "Estado"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Ticket médio por vendedor deste mês
SELECT "Vendedor",
       SUM("Valor") / NULLIF(COUNT(DISTINCT "Lancamento"), 0) AS ticket_medio,
       SUM("Valor")                                            AS faturamento_total,
       COUNT(DISTINCT "Lancamento")                            AS qtd_pedidos,
       SUM(SUM("Valor")) OVER ()                              AS total_geral_faturamento
FROM integralmix."fVendas"
WHERE DATE_TRUNC('month', "DataEmissao") = DATE_TRUNC('month', CURRENT_DATE)
GROUP BY "Vendedor"
ORDER BY ticket_medio DESC
LIMIT 20;

---

-- Top clientes por faturamento em janeiro de 2026
SELECT "Cliente",
       SUM("Valor")                 AS faturamento_total,
       COUNT(DISTINCT "Lancamento") AS qtd_pedidos,
       SUM(SUM("Valor")) OVER ()    AS total_geral_faturamento
FROM integralmix."fVendas"
WHERE "DataEmissao" >= '2026-01-01' AND "DataEmissao" < '2026-02-01'
GROUP BY "Cliente"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Melhor cliente em Fortaleza em janeiro de 2026
SELECT "Cliente",
       SUM("Valor")                 AS faturamento_total,
       COUNT(DISTINCT "Lancamento") AS qtd_pedidos,
       SUM(SUM("Valor")) OVER ()    AS total_geral_faturamento
FROM integralmix."fVendas"
WHERE "Cidade" ILIKE '%fortaleza%'
  AND "DataEmissao" >= '2026-01-01' AND "DataEmissao" < '2026-02-01'
GROUP BY "Cliente"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Comparação de faturamento mês atual vs mês anterior
SELECT TO_CHAR(DATE_TRUNC('month', data_emissao), 'MM/YYYY') AS periodo,
       SUM(faturamento)                                        AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
  AND data_emissao <  DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'
GROUP BY DATE_TRUNC('month', data_emissao)
ORDER BY DATE_TRUNC('month', data_emissao) ASC;

---

-- Evolução mensal do faturamento no ano de 2026
SELECT TO_CHAR(DATE_TRUNC('month', data_emissao), 'MM/YYYY') AS periodo,
       SUM(faturamento)                                        AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE EXTRACT(YEAR FROM data_emissao) = 2026
GROUP BY DATE_TRUNC('month', data_emissao)
ORDER BY DATE_TRUNC('month', data_emissao) ASC;

---

-- Evolução mensal dos últimos 6 meses
SELECT TO_CHAR(DATE_TRUNC('month', data_emissao), 'MM/YYYY') AS periodo,
       SUM(faturamento)                                        AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '5 months'
  AND data_emissao <  DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'
GROUP BY DATE_TRUNC('month', data_emissao)
ORDER BY DATE_TRUNC('month', data_emissao) ASC;"""


def _build_sql_prompt() -> str:
    """Monta o prompt SQL injetando apenas a data atual (parte dinâmica)."""
    hoje = datetime.now()
    return (
        _SQL_STATIC
        + f"\n\nHoje é {hoje.strftime('%Y-%m-%d')}. Ano vigente: {hoje.strftime('%Y')}."
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT C — Extrator de Card Data (JSON estruturado)
# Responsabilidade: extrair números reais dos resultados. Nunca inventar.
# ticket_medio removido daqui — calculado em Python após extração (melhoria #4).
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_CARD_EXTRACTOR = """Você é um extrator de dados estruturados.
Receberá o resultado bruto de uma consulta SQL (lista de dicionários Python) e a intenção da consulta.
Extraia os dados e retorne APENAS um JSON válido neste schema exato:

{
  "faturamento_total": <float ou null>,
  "qtd_pedidos": <int ou null>,
  "total_itens": <int ou null>,
  "variacao_ano_anterior": <float ou null>,
  "top_drivers": [
    {
      "nome": "<string>",
      "faturamento": <float>,
      "qtd_pedidos": <int ou null>
    }
  ],
  "serie_temporal": [
    {"periodo": "<MM/YYYY>", "faturamento": <float>}
  ]
}

REGRAS CRÍTICAS — TOLERÂNCIA ZERO PARA ALUCINAÇÃO:
- Se um campo NÃO está nos dados recebidos → use null. NUNCA invente valores.

- faturamento_total:
    PRIORIDADE 1: Se existir a coluna "total_geral_faturamento" nos dados, use esse valor.
    PRIORIDADE 2: Caso contrário, some todos os valores de "faturamento_total" ou "faturamento".
    EXCEÇÃO: Se for série temporal, use o valor do ÚLTIMO período (mais recente).

- qtd_pedidos:
    PRIORIDADE 1: Se existir "total_geral_pedidos", use esse valor.
    PRIORIDADE 2: Some os valores de pedidos das linhas, ou null se não existir.

- total_itens: número total de linhas retornadas pela query (len dos dados recebidos),
  independente do truncamento. Use sempre que houver agrupamento por entidade.
  Ex: se vieram 20 linhas de produtos → total_itens = 20. Se não há agrupamento → null.

- variacao_ano_anterior: se os dados contiverem duas colunas de faturamento — uma do período
  atual e uma do ano anterior (ex: "faturamento_atual" e "faturamento_anterior") — calcule
  (atual - anterior) / anterior. Caso contrário → null. NUNCA invente o valor anterior.

- serie_temporal: preencha quando os dados contêm coluna "periodo" com valores MM/YYYY
  e há 2 ou mais linhas de períodos de tempo.
  → Ordene do mais antigo ao mais recente.
  → Neste caso: top_drivers = [] e faturamento_total = valor do último período.

- top_drivers: se os dados têm agrupamento por ENTIDADE (produto, vendedor, estado, cidade,
  cliente — NÃO períodos de tempo), inclua TODOS os itens recebidos (até 20).
  "nome" = VALOR real da célula (ex: "Fortaleza", "João Silva"), NUNCA o nome da coluna.
  Se não há agrupamento por entidade ou é série temporal → use [].

- Retorne APENAS o JSON, sem markdown, sem explicações.
- NÃO inclua ticket_medio nem ticket_medio_item — calculados externamente."""


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT D — Correção de SQL (self-healing)
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_FIX = """Você é um especialista em SQL para PostgreSQL.
Receberá uma query SQL que falhou e a mensagem de erro do banco.
Corrija o SQL e retorne APENAS a string SQL corrigida, sem markdown, sem explicações."""


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: Python monta o card — zero chance de alucinação
# ─────────────────────────────────────────────────────────────────────────────

def _is_period_str(s: str) -> bool:
    """Verifica se uma string parece um período (MM/YYYY ou YYYY-MM-DD)."""
    import re
    return bool(re.match(r'^\d{2}/\d{4}$', str(s)) or re.match(r'^\d{4}-\d{2}', str(s)))


def _fmt_pct(value: float) -> str:
    """Formata percentual com sinal e uma casa decimal. Ex: +12,3% ou -4,1%"""
    return f"{value:+.1f}%".replace(".", ",")


def _var_arrow(pct: float) -> str:
    return "▲" if pct >= 0 else "▼"


def render_card(card_data: dict) -> str:
    """Monta o card Markdown a partir do JSON estruturado.
    Cada campo é exibido apenas se não for None — Python decide, não o LLM."""

    serie        = card_data.get("serie_temporal")       or []
    drivers      = card_data.get("top_drivers")          or []
    fat          = card_data.get("faturamento_total")
    ped          = card_data.get("qtd_pedidos")
    tkt          = card_data.get("ticket_medio")
    total_itens  = card_data.get("total_itens")
    var_yoy      = card_data.get("variacao_ano_anterior")

    # ── MODO SÉRIE TEMPORAL (3+ períodos) ────────────────────────────────────
    if len(serie) >= 3:
        total = sum(s.get("faturamento", 0) for s in serie)
        lines = [f"### 📈 Evolução de Faturamento\n**Total do período: {_fmt_brl(total)}**\n"]
        lines.append("| Período | Faturamento | Var. Mês |")
        lines.append("|:-------:|------------:|:--------:|")
        for i, s in enumerate(serie):
            v = s.get("faturamento", 0)
            if i == 0:
                var_str = "—"
            else:
                prev = serie[i - 1].get("faturamento", 0)
                if prev and prev > 0:
                    pct     = ((v - prev) / prev) * 100
                    var_str = f"{_var_arrow(pct)} {_fmt_pct(pct)}"
                else:
                    var_str = "—"
            lines.append(f"| {s['periodo']} | {_fmt_brl(v)} | {var_str} |")

        if serie[0].get("faturamento", 0) > 0:
            v0, vn = serie[0]["faturamento"], serie[-1]["faturamento"]
            trend  = ((vn - v0) / v0) * 100
            arrow  = "📈" if trend >= 0 else "📉"
            lines.append(f"\n{arrow} **Tendência {serie[0]['periodo']} → {serie[-1]['periodo']}: {_fmt_pct(trend)}**")
        return "\n".join(lines)

    # ── MODO COMPARAÇÃO (2 períodos via serie_temporal) ───────────────────────
    if len(serie) == 2:
        v_ant   = serie[0].get("faturamento", 0)
        v_atual = serie[1].get("faturamento", 0)
        lines   = ["### 📊 Comparativo de Períodos"]
        lines.append(f"- 🔵 {serie[0]['periodo']}: {_fmt_brl(v_ant)}")
        lines.append(f"- 🟢 {serie[1]['periodo']}: {_fmt_brl(v_atual)}")
        if v_ant > 0:
            var_pct = ((v_atual - v_ant) / v_ant) * 100
            diff    = v_atual - v_ant
            arrow   = "📈" if var_pct >= 0 else "📉"
            sinal   = "+" if diff >= 0 else ""
            lines.append(f"\n{arrow} Variação: **{_fmt_pct(var_pct)}** ({sinal}{_fmt_brl(diff)})")
        return "\n".join(lines)

    # ── MODO RELATÓRIO (drivers por entidade) ────────────────────────────────
    if drivers:
        fat_values = [d.get("faturamento") for d in drivers]

        # Fallback: drivers com períodos (comparação sem serie_temporal)
        if (len(drivers) == 2
                and all(v is not None for v in fat_values)
                and all(_is_period_str(d.get("nome", "")) for d in drivers)):
            log.warning(
                "render_card: fallback de comparação via top_drivers ativado. "
                "Verifique se o Prompt C está populando serie_temporal corretamente."
            )
            v_ant, v_atual = fat_values[0], fat_values[1]
            lines = ["### 📊 Comparativo de Períodos"]
            lines.append(f"- 🔵 {drivers[0]['nome']}: {_fmt_brl(v_ant)}")
            lines.append(f"- 🟢 {drivers[1]['nome']}: {_fmt_brl(v_atual)}")
            if v_ant and v_ant > 0:
                var_pct = ((v_atual - v_ant) / v_ant) * 100
                diff    = v_atual - v_ant
                arrow   = "📈" if var_pct >= 0 else "📉"
                sinal   = "+" if diff >= 0 else ""
                lines.append(f"\n{arrow} Variação: **{_fmt_pct(var_pct)}** ({sinal}{_fmt_brl(diff)})")
            return "\n".join(lines)

        # ── Card de KPIs globais (cabeçalho do relatório) ──────────────────
        lines = ["### 🎯 Visão Geral"]
        if fat is not None:
            fat_line = f"- 💰 **Faturamento Total:** {_fmt_brl(fat)}"
            if var_yoy is not None:
                arrow    = "📈" if var_yoy >= 0 else "📉"
                fat_line += f"  {arrow} *vs. ano anterior: {_fmt_pct(var_yoy)}*"
            lines.append(fat_line)
        if ped is not None:
            lines.append(f"- 📦 **Pedidos:** {_fmt_int(int(ped))}")
        if tkt is not None:
            lines.append(f"- 🎟️ **Ticket Médio:** {_fmt_brl(tkt)}")

        # ── Tabela Top 5 com participação, pedidos e ticket médio ─────────
        # Calcula participação com base no total real (fat ou soma dos drivers)
        base_fat = fat or sum(d.get("faturamento", 0) for d in drivers)

        # Cabeçalho dinâmico: inclui colunas apenas se houver dados
        has_pedidos = any(d.get("qtd_pedidos") is not None for d in drivers[:5])
        has_ticket  = any(d.get("ticket_medio_item") is not None for d in drivers[:5])

        top_label = "Top 5"
        if total_itens is not None and total_itens > 5:
            top_label = f"Top 5 de {_fmt_int(total_itens)}"

        lines.append(f"\n### 🏆 {top_label}")

        # Monta cabeçalho da tabela
        header = "| # | Item | Faturamento | Part.% |"
        sep    = "|:-:|:-----|------------:|-------:|"
        if has_pedidos:
            header += " Pedidos |"
            sep    += "--------:|"
        if has_ticket:
            header += " Ticket Médio |"
            sep    += "-------------:|"
        lines.append(header)
        lines.append(sep)

        medals = ["🥇", "🥈", "🥉", "4º", "5º"]
        for i, d in enumerate(drivers[:5]):
            medal  = medals[i] if i < len(medals) else f"{i+1}º"
            f_val  = d.get("faturamento") or 0
            part   = (f_val / base_fat * 100) if base_fat > 0 else 0
            p_val  = d.get("qtd_pedidos")
            tk_val = d.get("ticket_medio_item")

            row = f"| {medal} | {d['nome']} | {_fmt_brl(f_val)} | {part:.1f}% |"
            if has_pedidos:
                row += f" {_fmt_int(int(p_val)) if p_val is not None else '—'} |"
            if has_ticket:
                row += f" {_fmt_brl(tk_val) if tk_val is not None else '—'} |"
            lines.append(row)

        # Rodapé: concentração dos top 5
        if base_fat > 0 and len(drivers) >= 2:
            conc = sum(d.get("faturamento", 0) for d in drivers[:5]) / base_fat * 100
            lines.append(f"\n> 💡 Os top 5 representam **{conc:.1f}%** do faturamento total do período.")

        return "\n".join(lines)

    # ── MODO CARD EXECUTIVO SIMPLES (total sem agrupamento) ──────────────────
    lines = ["### 🎯 Resumo Executivo"]
    if fat is not None:
        fat_line = f"- 💰 **Faturamento:** {_fmt_brl(fat)}"
        if var_yoy is not None:
            arrow    = "📈" if var_yoy >= 0 else "📉"
            fat_line += f"  {arrow} *vs. ano anterior: {_fmt_pct(var_yoy)}*"
        lines.append(fat_line)
    if ped is not None:
        lines.append(f"- 📦 **Pedidos:** {_fmt_int(int(ped))}")
    if tkt is not None:
        lines.append(f"- 🎟️ **Ticket Médio:** {_fmt_brl(tkt)}")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS INTERNOS
# ─────────────────────────────────────────────────────────────────────────────

def _rows_limit_for_intent(standalone_intent: str) -> int:
    """Melhoria #5: envia menos rows ao Prompt C quando a query é um total simples.
    Queries de ranking/série se beneficiam de mais linhas; totais simples não precisam."""
    intent_lower = standalone_intent.lower()
    ranking_signals = ("top", "ranking", "maiores", "melhores", "por produto", "por cliente",
                       "por vendedor", "por estado", "por cidade", "por região", "evolução",
                       "mês a mês", "série", "comparação")
    if any(signal in intent_lower for signal in ranking_signals):
        return 20
    return 5


def _calculate_ticket_medio(card_data: dict) -> dict:
    """Calcula ticket_medio global e ticket_medio_item por driver, tudo em Python."""
    fat = card_data.get("faturamento_total")
    ped = card_data.get("qtd_pedidos")
    card_data["ticket_medio"] = (fat / ped) if (fat and ped and ped > 0) else None

    for d in card_data.get("top_drivers") or []:
        f_val = d.get("faturamento")
        p_val = d.get("qtd_pedidos")
        d["ticket_medio_item"] = (f_val / p_val) if (f_val and p_val and p_val > 0) else None

    return card_data


# ─────────────────────────────────────────────────────────────────────────────
# API PÚBLICA
# ─────────────────────────────────────────────────────────────────────────────

def resolve_intent(history: list[dict]) -> str:
    """Passo 1: Roteia e resolve o intent do usuário a partir do histórico.
    Retorna RESPOSTA_DIRETA:, CLARIFICACAO: ou uma frase standalone limpa.
    Usa MODEL_ROUTER (gpt-4o) para maior precisão no roteamento."""
    resolved = client.chat.completions.create(
        model=MODEL_ROUTER,
        messages=[
            {"role": "system", "content": _build_resolver_prompt()},
            *history,
        ],
        temperature=0,
    ).choices[0].message.content.strip()

    # Melhoria #8: log estruturado da intent resolvida para diagnóstico em produção
    last_user_msg = next(
        (m["content"] for m in reversed(history) if m.get("role") == "user"), ""
    )
    log.info(
        "intent_resolved",
        extra={"intent": resolved, "user_msg": last_user_msg},
    )
    return resolved


def generate_sql(standalone_intent: str) -> str:
    """Passo 2: Gera SQL a partir de uma instrução standalone já resolvida.
    Não recebe histórico — sem contaminação de contexto."""
    return client.chat.completions.create(
        model=MODEL_SQL,
        messages=[
            {"role": "system", "content": _build_sql_prompt()},
            {"role": "user", "content": standalone_intent},
        ],
        temperature=0,
    ).choices[0].message.content.strip()


def fix_sql(broken_sql: str, db_error: str) -> str:
    """Melhoria #2: self-healing com retry. Levanta RuntimeError explícito se falhar,
    permitindo que o caller exiba mensagem amigável ao usuário."""
    log.warning(
        "fix_sql: tentando auto-correção. Erro original: %s",
        db_error,
    )
    fixed = client.chat.completions.create(
        model=MODEL_FIX,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT_FIX},
            {"role": "user", "content": f"SQL com erro:\n{broken_sql}\n\nErro:\n{db_error}"},
        ],
        temperature=0,
    ).choices[0].message.content.strip()

    if not fixed or fixed.upper().startswith("PERGUNTA_INVALIDA"):
        raise RuntimeError(
            "Não foi possível corrigir a query automaticamente. "
            "Por favor, reformule a pergunta ou entre em contato com o suporte."
        )
    return fixed


def extract_card_data(standalone_intent: str, rows: list) -> dict:
    """Passo 3a: LLM extrai JSON estruturado dos resultados brutos.
    Melhorias aplicadas:
      - #5: limite de rows adaptado ao tipo de query (ranking vs. total simples)
      - #4: ticket_medio calculado em Python após extração, não pelo LLM
    Usa json_object mode — garante JSON válido na saída."""
    max_rows = _rows_limit_for_intent(standalone_intent)
    safe_rows = rows[:max_rows]

    if len(rows) > max_rows:
        log.info(
            "extract_card_data: payload truncado %d → %d linhas (intent: %s).",
            len(rows), max_rows, standalone_intent,
        )

    response = client.chat.completions.create(
        model=MODEL_CARD,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT_CARD_EXTRACTOR},
            {"role": "user", "content": f"Intenção: {standalone_intent}\n\nDados: {safe_rows}"},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    raw = response.choices[0].message.content.strip()
    try:
        card_data = json.loads(raw)
    except json.JSONDecodeError:
        log.error("extract_card_data: falha ao parsear JSON do extrator: %s", raw)
        return {}

    # Melhoria #4: ticket_medio calculado aqui, nunca pelo LLM
    return _calculate_ticket_medio(card_data)