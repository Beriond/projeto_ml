# Home Credit Default Risk: Automação e Gestão de Risco

Este projeto foi desenvolvido como requisito final do módulo de Inteligência Artificial e Machine Learning da **LabData FIA**. O objetivo central é otimizar a concessão de crédito através de modelos preditivos, equilibrando a proteção contra a inadimplência com a eficiência na aprovação.

## 🎯 Desafio de Negócio
O Home Credit atende clientes com histórico de crédito limitado. O desafio é transformar o processo de análise de crédito em um fluxo automatizado, ágil e matematicamente preciso.
* **Problema:** Alta exposição à inadimplência e perda de oportunidade (recusa de bons clientes).
* **Solução:** Implementação de um pipeline de Machine Learning (*Gradient Boosting*) com calibração estratégica de *threshold* para controle de apetite ao risco.

## 🛠 Arquitetura e Documentação Técnica
O projeto prioriza a qualidade do código e a robustez do fluxo de dados:

1. **Pipeline de Sanitização (ETL):** - Tratamento de valores sentinela (ex: correção do valor `365243` em `DAYS_EMPLOYED`).
   - Feature Engineering: Conversão de unidades de tempo para facilitar a interpretabilidade.
   - Validação de integridade: Checagem de valores válidos para o `TARGET` no conjunto de treino.
   - Portabilidade: Uso de `pathlib` e configurações via `YAML` para evitar *hardcoding*.

2. **Engenharia de Variáveis:**
   - Criação de tabelas analíticas (ABT) estruturadas para facilitar o consumo pelos algoritmos.
   - Remoção de duplicatas e otimização de tipos de dados para performance.

3. **Treinamento e Gestão de Experimentos:**
   - Treinamento paralelo de múltiplos algoritmos (`Random Forest`, `Decision Tree`, `HistGradientBoosting`).
   - Persistência de artefatos (`.pkl`) e documentação gráfica da importância das variáveis (`importance.png`).

## 🚀 Guia de Execução

### 1. Configuração do Ambiente
```bash
# Clone e instale as dependências
git clone [seu-repositorio]
python -m venv .venv
source .venv/bin/activate  # ou .\.venv\Scripts\Activate.ps1 no Windows

pip install -r requirements.txt

2. Preparação dos Dados
Coloque os arquivos do Kaggle na pasta Dados/:
- application_train.csv, bureau.csv, previous_application.csv.

3. Pipeline de Execução (Ordem)
# 1. Sanitização (Dados brutos -> Limpos)
python DataPipeline/data_sanitization2.py

# 2. Engenharia de Variáveis (Preparação da base de treino)
python DataPipeline/abt_transform.py

# 3. Treinamento (Geração do modelo campeão)
python Model/train.py

📊 Metodologia de Avaliação
O modelo utiliza métricas robustas para garantir performance e alinhamento com o negócio:
- ROC AUC: Mede o poder discriminatório do modelo independente do ponto de corte.
- Índice de Youden: Critério estatístico utilizado para encontrar o threshold ideal que equilibra Sensibilidade e Especificidade.
- Matriz de Confusão: Ferramenta visual para análise dos tipos de erro (Falsos Positivos vs. Falsos Negativos).

O sistema permite que a gestão altere o Threshold de Decisão via arquivo Model/config.yml, permitindo alternar entre estratégias de segurança (aversão ao risco) ou crescimento (volume de aprovações de crédito).