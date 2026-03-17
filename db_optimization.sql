-- =============================================================================
-- db_optimization.sql
-- Camada Semântica de Agregação — IntegralMix
-- Rodar uma vez no PostgreSQL. Após rodar, executar REFRESH periodicamente.
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. MATERIALIZED VIEW: agg_vendas_diarias
--    Pré-agrega métricas por dia + dimensões principais.
--    Reduz o volume de leitura de 1.2M → ~50k linhas típicas.
-- -----------------------------------------------------------------------------
DROP MATERIALIZED VIEW IF EXISTS integralmix.agg_vendas_diarias;

CREATE MATERIALIZED VIEW integralmix.agg_vendas_diarias AS
SELECT
    -- Dimensão temporal
    DATE_TRUNC('day', "DataEmissao")::date          AS data_emissao,
    DATE_TRUNC('month', "DataEmissao")::date         AS mes_emissao,

    -- Dimensões de empresa
    "CodigoEmpresa",
    "Empresa",
    "EmpresaGerencial",

    -- Dimensões de cliente
    "CodigoCliente",
    "Cliente",

    -- Dimensões de produto
    "CodigoObjeto",
    "Objeto",
    "CodigoObjetoMae",
    "ObjetoMae",

    -- Dimensões de vendedor / hierarquia comercial
    "CodigoVendedor",
    "Vendedor",
    "Gerente",
    "Supervisor",

    -- Dimensões geográficas e comerciais
    "Estado",
    "Cidade",
    "CanalCliente",
    "Segmento",
    "AreaVenda",
    "AreaVendaMae",
    "FormadePagamento",
    "TipodeDocumento",

    -- Métricas pré-calculadas
    SUM("Valor")                                                        AS faturamento,
    COUNT(DISTINCT "Lancamento")                                        AS qtd_pedidos,
    COUNT(DISTINCT "CodigoCliente")                                     AS qtd_clientes,
    COUNT(DISTINCT "CodigoObjeto")                                      AS qtd_produtos,
    SUM("Qtd") / 1000.0                                                 AS volume_toneladas,
    SUM("Valor") / NULLIF(COUNT(DISTINCT "Lancamento"), 0)              AS ticket_medio

FROM integralmix."fVendas"
GROUP BY
    DATE_TRUNC('day',   "DataEmissao")::date,
    DATE_TRUNC('month', "DataEmissao")::date,
    "CodigoEmpresa", "Empresa", "EmpresaGerencial",
    "CodigoCliente", "Cliente",
    "CodigoObjeto",  "Objeto", "CodigoObjetoMae", "ObjetoMae",
    "CodigoVendedor", "Vendedor", "Gerente", "Supervisor",
    "Estado", "Cidade", "CanalCliente", "Segmento",
    "AreaVenda", "AreaVendaMae", "FormadePagamento", "TipodeDocumento"
WITH DATA;


-- -----------------------------------------------------------------------------
-- 2. ÍNDICES — colunas mais usadas em WHERE / GROUP BY
-- -----------------------------------------------------------------------------

-- Temporal (mais críticos)
CREATE INDEX idx_agg_data_emissao  ON integralmix.agg_vendas_diarias (data_emissao);
CREATE INDEX idx_agg_mes_emissao   ON integralmix.agg_vendas_diarias (mes_emissao);

-- Dimensões de negócio
CREATE INDEX idx_agg_empresa       ON integralmix.agg_vendas_diarias ("CodigoEmpresa");
CREATE INDEX idx_agg_cliente       ON integralmix.agg_vendas_diarias ("CodigoCliente");
CREATE INDEX idx_agg_objeto        ON integralmix.agg_vendas_diarias ("CodigoObjeto");
CREATE INDEX idx_agg_vendedor      ON integralmix.agg_vendas_diarias ("CodigoVendedor");
CREATE INDEX idx_agg_estado        ON integralmix.agg_vendas_diarias ("Estado");
CREATE INDEX idx_agg_segmento      ON integralmix.agg_vendas_diarias ("Segmento");


-- -----------------------------------------------------------------------------
-- 3. REFRESH (rodar diariamente após a ingestão)
--    Descomentar e agendar no Airflow / cron:
-- -----------------------------------------------------------------------------
-- REFRESH MATERIALIZED VIEW CONCURRENTLY integralmix.agg_vendas_diarias;


-- -----------------------------------------------------------------------------
-- 4. Verificação rápida
-- -----------------------------------------------------------------------------
SELECT
    COUNT(*)                        AS total_linhas,
    MIN(data_emissao)               AS data_min,
    MAX(data_emissao)               AS data_max,
    ROUND(SUM(faturamento)::numeric, 2) AS faturamento_total
FROM integralmix.agg_vendas_diarias;
