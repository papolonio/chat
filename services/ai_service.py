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
  - Coluna de valor:  "Valor"      (com aspas, V maiúsculo)
  - Coluna de data:   "DataEmissao"(com aspas, D e E maiúsculos)
  - ⚠ Não existe coluna faturamento (sem aspas) nesta tabela.

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
    Para contar pedidos distintos use fVendas com COUNT(DISTINCT "Lancamento").

▶ integralmix."fVendas" — PARA PEDIDOS E TICKET MÉDIO
  Use quando precisar de COUNT(DISTINCT "Lancamento") ou detalhes de linha individual.
  Coluna de valor: "Valor" (com aspas, V maiúsculo). Coluna de data: "DataEmissao".
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
GLOSSÁRIO DE NEGÓCIO — Termos do usuário → coluna/tabela
════════════════════════════════════════
Use este glossário para interpretar o que o usuário quer dizer:

▶ DIMENSÕES DE PRODUTO (requer JOIN com INT_dim_Objetos):
  "linha do produto" / "linha"       → dim."LinhaObjeto"
  "divisão do produto" / "divisão"   → dim."DivisaoObjeto"
  "descrição gerencial"              → dim."DescricaoGerencial"
  "tipo de produto"                  → dim."TipoObjeto"
  "produto ativo" / "apenas ativos"  → dim."Ativo" = true

▶ DIMENSÕES DE CLIENTE (requer JOIN com GR_dim_Clientes):
  "razão social" / "nome do cliente" → cli."RazaoSocial"
  "divisão do cliente"               → cli."DivisaoCliente"
  "tipo de estabelecimento" / "tipo de cliente" → cli."TipoEstabelecimento"
  "grupo empresarial" / "grupo"      → cli."Grupo_Empresa"
  "município do cliente"             → cli."Municipio"

▶ DIMENSÕES DISPONÍVEIS EM agg_vendas_diarias (sem JOIN):
  "canal" / "canal de venda" / "canal de atendimento" → "CanalCliente"
    (valores típicos: Distribuidor, Cooperativa, Varejo, Indústria, Produtor Rural)
  "segmento" / "segmento do cliente"                  → "Segmento"
  "área de venda" / "área"                            → "AreaVenda"
  "área mãe" / "área gerencial"                       → "AreaVendaMae"
  "forma de pagamento" / "pagamento"                  → "FormadePagamento"
  "tipo de documento" / "tipo de nota"                → "TipodeDocumento"
  "empresa" / "filial"                                → "Empresa"
  "empresa gerencial" / "grupo empresa"               → "EmpresaGerencial"
  "gerente" / "gerente de vendas"                     → "Gerente"
  "supervisor" / "supervisor de vendas"               → "Supervisor"
  "volume" / "toneladas" / "peso"                     → volume_toneladas  (em toneladas)
  "produto pai" / "família de produto"                → "ObjetoMae"

▶ MÉTRICAS:
  "faturamento" / "receita" / "vendas" / "quanto vendeu" → SUM(faturamento) ou SUM("Valor")
  "ticket médio" / "valor médio por pedido"              → SUM("Valor") / COUNT(DISTINCT "Lancamento")
  "volume de pedidos" / "quantidade de pedidos" / "pedidos" → COUNT(DISTINCT "Lancamento") via fVendas
  "clientes atendidos" / "base de clientes"              → COUNT(DISTINCT "CodigoCliente")
  "mix de produtos" / "produtos distintos"               → COUNT(DISTINCT "CodigoObjeto")

════════════════════════════════════════
REGRAS DE SAÍDA
════════════════════════════════════════
1. Retorne APENAS a string SQL pura. Sem markdown, sem ```sql, sem explicações.
2. Aspas duplas em colunas com maiúsculas. Colunas minúsculas da view não precisam.
3. NUNCA gere INSERT, UPDATE, DELETE, DROP, CREATE ou qualquer DDL/DML.
4. Filtros de texto: SEMPRE use ILIKE com % (ex: WHERE "Cliente" ILIKE '%nome%').
   ALIAS DE DIMENSÕES (palavras do usuário → coluna real):
   "região" / "regiões" / "regional" → coluna "Estado"
   "produto" / "produtos" / "item" / "itens" → coluna "Objeto"
   "vendedor" / "representante" → coluna "Vendedor"
   "cliente" / "clientes" / "comprador" → coluna "Cliente"
   "cidade" / "município" → coluna "Cidade"
5. MAPEAMENTO DE INTENÇÕES — estas perguntas NUNCA devem retornar PERGUNTA_INVALIDA:
   - "produtos vendidos", "quais produtos", "top produtos" → GROUP BY "Objeto"
   - "melhores clientes", "top clientes", "quais clientes" → GROUP BY "Cliente"
   - "vendedores", "top vendedores"                       → GROUP BY "Vendedor"
   - "por região", "por estado", "regiões"                → GROUP BY "Estado"
   - "por cidade", "cidades"                              → GROUP BY "Cidade"
   Só retorne PERGUNTA_INVALIDA se a pergunta exigir tabelas ou dados que não existem.
6. 🔴 REPORT MODE: Para perguntas amplas sem filtro de entidade específica, NUNCA retorne
   apenas um SUM() isolado. Use GROUP BY pela dimensão mais relevante ("Objeto", "Estado"
   ou "Vendedor") + sempre inclua SUM("Valor") AS faturamento_total + COUNT(DISTINCT "Lancamento")
   AS qtd_pedidos, ORDER BY faturamento_total DESC LIMIT 20.

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

-- Instrução: "Produtos vendidos em janeiro de 2026" ou "top produtos de janeiro de 2026"
SELECT "Objeto",
       SUM(faturamento)                          AS faturamento_total,
       SUM(SUM(faturamento)) OVER ()             AS total_geral_faturamento
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= '2026-01-01' AND data_emissao < '2026-02-01'
GROUP BY "Objeto"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Instrução: "Faturamento total de março de 2026" (sem pedidos — usa agg_vendas_diarias)
SELECT SUM(faturamento)                          AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= '2026-03-01' AND data_emissao < '2026-04-01';

---

-- Instrução: "Faturamento por estado dos últimos 15 dias" (sem pedidos — usa agg_vendas_diarias)
SELECT "Estado",
       SUM(faturamento)                          AS faturamento_total,
       SUM(SUM(faturamento)) OVER ()             AS total_geral_faturamento
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= CURRENT_DATE - INTERVAL '15 days'
GROUP BY "Estado"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Instrução: "Relatório de faturamento de março de 2026 agrupado por produto" (sem pedidos — usa agg_vendas_diarias)
SELECT "Objeto",
       SUM(faturamento)                          AS faturamento_total,
       SUM(SUM(faturamento)) OVER ()             AS total_geral_faturamento
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= '2026-03-01' AND data_emissao < '2026-04-01'
GROUP BY "Objeto"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Instrução: "Faturamento total de fevereiro de 2026 com contagem de pedidos" (usa fVendas)
SELECT SUM("Valor")                              AS faturamento_total,
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
       SUM(SUM("Valor")) OVER ()                               AS total_geral_faturamento
FROM integralmix."fVendas"
WHERE DATE_TRUNC('month', "DataEmissao") = DATE_TRUNC('month', CURRENT_DATE)
GROUP BY "Vendedor"
ORDER BY ticket_medio DESC
LIMIT 20;

---

-- Instrução: "Top clientes por faturamento em janeiro de 2026"
SELECT "Cliente",
       SUM("Valor")                AS faturamento_total,
       COUNT(DISTINCT "Lancamento") AS qtd_pedidos,
       SUM(SUM("Valor")) OVER ()   AS total_geral_faturamento
FROM integralmix."fVendas"
WHERE "DataEmissao" >= '2026-01-01' AND "DataEmissao" < '2026-02-01'
GROUP BY "Cliente"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Instrução: "Melhor cliente em Fortaleza em janeiro de 2026"
SELECT "Cliente",
       SUM("Valor")                AS faturamento_total,
       COUNT(DISTINCT "Lancamento") AS qtd_pedidos,
       SUM(SUM("Valor")) OVER ()   AS total_geral_faturamento
FROM integralmix."fVendas"
WHERE "Cidade" ILIKE '%fortaleza%'
  AND "DataEmissao" >= '2026-01-01' AND "DataEmissao" < '2026-02-01'
GROUP BY "Cliente"
ORDER BY faturamento_total DESC
LIMIT 20;

---

-- Instrução: "Comparação de faturamento entre o mês atual e o mês anterior"
-- Instrução: "e em comparação ao mês anterior?" / "evolução mês a mês" / "variação vs mês passado"
SELECT TO_CHAR(DATE_TRUNC('month', data_emissao), 'MM/YYYY') AS periodo,
       SUM(faturamento)                                        AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '1 month'
  AND data_emissao <  DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'
GROUP BY DATE_TRUNC('month', data_emissao)
ORDER BY DATE_TRUNC('month', data_emissao) ASC;

---

-- Instrução: "Comparação de faturamento de janeiro de 2026 com dezembro de 2025"
SELECT TO_CHAR(DATE_TRUNC('month', data_emissao), 'MM/YYYY') AS periodo,
       SUM(faturamento)                                        AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= '2025-12-01' AND data_emissao < '2026-02-01'
GROUP BY DATE_TRUNC('month', data_emissao)
ORDER BY DATE_TRUNC('month', data_emissao) ASC;

---

-- Instrução: "Evolução mensal do faturamento no ano de 2026" / "mês a mês de 2026" / "histórico 2026"
SELECT TO_CHAR(DATE_TRUNC('month', data_emissao), 'MM/YYYY') AS periodo,
       SUM(faturamento)                                        AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE EXTRACT(YEAR FROM data_emissao) = 2026
GROUP BY DATE_TRUNC('month', data_emissao)
ORDER BY DATE_TRUNC('month', data_emissao) ASC;

---

-- Instrução: "Evolução mensal dos últimos 6 meses" / "série dos últimos 6 meses"
SELECT TO_CHAR(DATE_TRUNC('month', data_emissao), 'MM/YYYY') AS periodo,
       SUM(faturamento)                                        AS faturamento_total
FROM integralmix.agg_vendas_diarias
WHERE data_emissao >= DATE_TRUNC('month', CURRENT_DATE) - INTERVAL '5 months'
  AND data_emissao <  DATE_TRUNC('month', CURRENT_DATE) + INTERVAL '1 month'
GROUP BY DATE_TRUNC('month', data_emissao)
ORDER BY DATE_TRUNC('month', data_emissao) ASC;
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
    EXCEÇÃO: Se for série temporal (ver abaixo), use o valor do ÚLTIMO período (mais recente).

- qtd_pedidos:
    PRIORIDADE 1: Se existir "total_geral_pedidos", use esse valor.
    PRIORIDADE 2: Some os valores de pedidos das linhas, ou null se não existir.

- ticket_medio: calcule faturamento_total / qtd_pedidos APENAS se ambos existirem; caso contrário null.

- serie_temporal: preencha quando os dados contêm uma coluna "periodo" com valores de mês/ano
  (ex: "01/2026", "02/2026") E há 2 ou mais linhas — cada uma é um período de tempo.
  → Ordene do mais antigo ao mais recente (ordem cronológica crescente).
  → "periodo" deve ser o valor da coluna "periodo" tal como está nos dados.
  → Neste caso: top_drivers = [] e faturamento_total = valor do período mais recente.
  → Se os dados NÃO têm coluna "periodo", use null.

- top_drivers: se os dados têm agrupamento por ENTIDADE (produto, vendedor, estado, cidade,
  cliente — NÃO períodos de tempo), inclua os 3 primeiros itens por maior faturamento.
  "nome" = VALOR real da célula (ex: "Fortaleza", "João Silva"), NUNCA o nome da coluna.
  Se não há agrupamento por entidade ou é série temporal, use [].

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

def _is_period_str(s: str) -> bool:
    """Verifica se uma string parece um período (MM/YYYY ou YYYY-MM-DD)."""
    import re
    return bool(re.match(r'^\d{2}/\d{4}$', str(s)) or re.match(r'^\d{4}-\d{2}', str(s)))


def render_card(card_data: dict) -> str:
    """Monta o card Markdown a partir do JSON estruturado.
    Cada campo é exibido apenas se não for None — Python decide, não o LLM."""

    serie   = card_data.get("serie_temporal") or []
    drivers = card_data.get("top_drivers")    or []
    fat     = card_data.get("faturamento_total")
    ped     = card_data.get("qtd_pedidos")
    tkt     = card_data.get("ticket_medio")

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
                    pct = ((v - prev) / prev) * 100
                    arrow = "▲" if pct >= 0 else "▼"
                    var_str = f"{arrow} {pct:+.1f}%"
                else:
                    var_str = "—"
            lines.append(f"| {s['periodo']} | {_fmt_brl(v)} | {var_str} |")

        # Tendência geral (primeiro → último)
        if serie[0].get("faturamento", 0) > 0:
            v0, vn = serie[0]["faturamento"], serie[-1]["faturamento"]
            trend  = ((vn - v0) / v0) * 100
            arrow  = "📈" if trend >= 0 else "📉"
            lines.append(f"\n{arrow} **Tendência {serie[0]['periodo']} → {serie[-1]['periodo']}: {trend:+.1f}%**")
        return "\n".join(lines)

    # ── MODO COMPARAÇÃO (2 períodos via serie_temporal) ───────────────────────
    if len(serie) == 2:
        v_ant   = serie[0].get("faturamento", 0)
        v_atual = serie[1].get("faturamento", 0)
        lines = ["### 📊 Comparativo de Períodos"]
        lines.append(f"- 🔵 {serie[0]['periodo']}: {_fmt_brl(v_ant)}")
        lines.append(f"- 🟢 {serie[1]['periodo']}: {_fmt_brl(v_atual)}")
        if v_ant > 0:
            var_pct = ((v_atual - v_ant) / v_ant) * 100
            diff    = v_atual - v_ant
            arrow   = "📈" if var_pct >= 0 else "📉"
            sinal   = "+" if diff >= 0 else ""
            lines.append(f"\n{arrow} Variação: **{var_pct:+.1f}%** ({sinal}{_fmt_brl(diff)})")
        return "\n".join(lines)

    # ── MODO CARD EXECUTIVO (entidades / total simples) ───────────────────────
    lines = ["### 🎯 Resumo Executivo"]
    if fat is not None:
        lines.append(f"- 💰 Faturamento: {_fmt_brl(fat)}")
    if ped is not None:
        lines.append(f"- 📦 Volume: {_fmt_int(int(ped))} pedidos")
    if tkt is not None:
        lines.append(f"- 🎟️ Ticket Médio: {_fmt_brl(tkt)}")

    if drivers:
        # Fallback: detecta se drivers são períodos (comparação sem serie_temporal)
        fat_values = [d.get("faturamento") for d in drivers]
        if (len(drivers) == 2
                and all(v is not None for v in fat_values)
                and all(_is_period_str(d.get("nome", "")) for d in drivers)):
            v_ant, v_atual = fat_values[0], fat_values[1]
            lines.append("\n### 📊 Comparativo de Períodos")
            lines.append(f"- 🔵 {drivers[0]['nome']}: {_fmt_brl(v_ant)}")
            lines.append(f"- 🟢 {drivers[1]['nome']}: {_fmt_brl(v_atual)}")
            if v_ant and v_ant > 0:
                var_pct = ((v_atual - v_ant) / v_ant) * 100
                diff    = v_atual - v_ant
                arrow   = "📈" if var_pct >= 0 else "📉"
                sinal   = "+" if diff >= 0 else ""
                lines.append(f"\n{arrow} Variação: **{var_pct:+.1f}%** ({sinal}{_fmt_brl(diff)})")
        else:
            medals = ["🥇 1º", "🥈 2º", "🥉 3º"]
            lines.append("\n### 🏆 Top 3 Drivers")
            for i, d in enumerate(drivers[:3]):
                medal = medals[i] if i < len(medals) else f"{i + 1}º"
                f_val = d.get("faturamento")
                p_val = d.get("qtd_pedidos")
                if f_val is not None:
                    lines.append(f"{medal} - {d['nome']} - {_fmt_brl(f_val)}")
                elif p_val is not None:
                    lines.append(f"{medal} - {d['nome']} - {_fmt_int(int(p_val))} pedidos")
                else:
                    lines.append(f"{medal} - {d['nome']}")

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
