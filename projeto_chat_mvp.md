Master Plan: MVP Chat com Dados (Text-to-SQL)
1. Visão Geral
O objetivo é construir um chat com IA (usando a API da OpenAI) que responda a perguntas de negócios consultando um banco de dados PostgreSQL. O sistema precisa ser simples, elegante, com uma interface bonita e estar pronto para produção em 24h.

2. Stack Tecnológica
Backend: Python 3 + Flask

Banco de Dados: PostgreSQL (schema integralmix, tabela fVendas)

IA: OpenAI API (modelo gpt-4o-mini ou gpt-3.5-turbo para velocidade/custo)

Frontend: HTML/JS padrão + Tailwind CSS (via CDN para design moderno e responsivo)

Abordagem IA: Text-to-SQL direto (Backend recebe a pergunta, envia schema para LLM, LLM devolve SQL, Backend roda no Postgres e devolve resultado).

3. Dicionário de Dados e Regras de Negócio
O sistema consultará uma única tabela desnormalizada chamada fVendas.

Colunas Disponíveis:
CodigoEmpresa, Empresa, CodigoObjeto, Objeto, Numero, CodigoCliente, Cliente, CodigoObjetoMae, ObjetoMae, CodigoVendedor, Vendedor, Gerente, Supervisor, CanalCliente, Estado, Cidade, Bairro, TipoLogradouro, Rua, Nr, TipoPessoa, DataEmissao, DataContabil, TipodeDocumento, CodigoTipodeOperacao, NomedoTipodeOperacao, CEP, Lancamento, CodigoPessoaEmpresa, CodigoPessoaCliente, Qtd, Valor, EmpresaGerencial, Latitude, Longitude, AreaVendaMae, AreaVenda, Segmento, CentroCusto, Prazo, FormadePagamento, Totalizador, CodigoPedidoDeVenda, CodigoCentroCusto, QtdSacos.

Métricas Essenciais (O LLM deve ser instruído a usar essas lógicas no SQL):

Faturamento: SUM(Valor)

Quantidade Pedidos: COUNT(DISTINCT Lancamento)

Ticket Médio: SUM(Valor) / COUNT(DISTINCT Lancamento)

Quantidade Clientes: COUNT(DISTINCT CodigoCliente)

Quantidade Produtos: COUNT(DISTINCT CodigoObjeto)

Volume Total (t): SUM(Qtd) / 1000.0

4. Regras de Arquitetura (Instruções para o Builder)
O código deve ser modular: separe as rotas (Flask), o serviço de IA (OpenAI) e o serviço de Banco (SQLAlchemy/psycopg2).

Zero credenciais no código. Tudo via .env.

A interface deve simular um chat moderno (estilo ChatGPT), com balões de mensagens, área de input clara e estado de "digitando/pensando".

Sempre use consultas READ ONLY. O banco de dados nunca pode ser alterado pela IA.
[FIM DO ARQUIVO MD]