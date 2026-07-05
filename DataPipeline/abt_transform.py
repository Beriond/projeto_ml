"""
Módulo de Construção da Analytical Base Table (ABT).

Este script transforma dados brutos (Application, Bureau, Previous Application) em 
uma base consolidada de modelagem (ABT), focando em criar variáveis que capturam 
o comportamento financeiro e o histórico de crédito do solicitante.

Pipeline de Transformação:
1. Agregação Temporal: Resume o histórico de empréstimos anteriores e bureau.
2. Feature Engineering: Cria proxies para "risco" e "urgência financeira".
3. Estratificação: Garante que a distribuição de bons/maus pagadores seja 
   mantida entre treino e validação (essencial para evitar bias).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "DataPipeline" / "config.yml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    """Carrega o arquivo de configuração YAML do projeto."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def build_abt(cfg: dict | None = None) -> None:
    """
    Constrói a ABT consolidando métricas comportamentais de crédito.
    
    A ABT é salva em dois arquivos (treino e validação) para garantir a 
    consistência durante o treinamento do modelo.
    """
    cfg = cfg or load_config()
    clean_dir = PROJECT_ROOT / cfg["paths"]["clean_dir"]
    abt_dir = PROJECT_ROOT / cfg["paths"]["abt_dir"]
    abt_dir.mkdir(parents=True, exist_ok=True)

    target = cfg["project"]["target"]

    print("--- Carregando bases de dados ---")
    app = pd.read_csv(clean_dir / cfg["data"]["clean_files"]["application"])
    prev = pd.read_csv(clean_dir / cfg["data"]["clean_files"]["previous_application"])
    bur = pd.read_csv(clean_dir / cfg["data"]["clean_files"]["bureau"])

    # 1. Indicadores de Histórico (Proxy de Lealdade e Aprovação)
    # Motivação: Clientes com histórico de aprovação consistente tendem a ter menos risco.
    prev['IS_APPROVED'] = (prev['NAME_CONTRACT_STATUS'] == 'Approved').astype(int)
    prev_stats = prev.groupby('SK_ID_CURR').agg({
        'SK_ID_PREV': 'count',
        'IS_APPROVED': 'mean',
        'DAYS_DECISION': 'max'
    }).rename(columns={'SK_ID_PREV': 'PREV_COUNT', 'IS_APPROVED': 'PREV_APPROVAL_RATE'})

    # 2. Indicadores de Bureau (Proxy de Desespero Financeiro)
    # Motivação: Aumento súbito de novos empréstimos (-90 dias) é um sinal de alerta (RED FLAG).
    bur['IS_RECENT_LOAN'] = (bur['DAYS_CREDIT'] > -90).astype(int)
    bureau_stats = bur.groupby('SK_ID_CURR').agg({
        'SK_ID_BUREAU': 'count',
        'AMT_CREDIT_SUM_DEBT': 'sum',
        'AMT_CREDIT_SUM': 'sum',
        'CREDIT_DAY_OVERDUE': 'max',
        'IS_RECENT_LOAN': 'sum',
        'DAYS_CREDIT': 'min'
    }).rename(columns={'SK_ID_BUREAU': 'BUREAU_LOAN_COUNT', 'CREDIT_DAY_OVERDUE': 'MAX_OVERDUE_DAYS'})

    # 3. Engenharia de Features
    # Unificamos as bases para criar indicadores normalizados.
    df = app.merge(prev_stats, on='SK_ID_CURR', how='left').merge(bureau_stats, on='SK_ID_CURR', how='left')
    
    def add_behavioral_features(data: pd.DataFrame) -> pd.DataFrame:
        """Cria variáveis de engenharia que normalizam o risco do cliente."""
        # URGENCY_SCORE: Alta taxa de empréstimos recentes indica busca desesperada por liquidez.
        data['URGENCY_SCORE'] = data['IS_RECENT_LOAN'] / data['BUREAU_LOAN_COUNT'].replace(0, np.nan)
        
        # OVERDUE_RISK_INDEX: Pior atraso histórico ajustado pelo tempo de relacionamento.
        data['OVERDUE_RISK_INDEX'] = data['MAX_OVERDUE_DAYS'] / (data['DAYS_CREDIT'].abs() / 365).replace(0, np.nan)
        
        # TOTAL_DEBT_TO_INCOME: Mede a capacidade de pagamento (DTI).
        data['TOTAL_DEBT_TO_INCOME'] = data['AMT_CREDIT_SUM_DEBT'] / data['AMT_INCOME_TOTAL'].replace(0, np.nan)
        
        return data.fillna(0)

    df = add_behavioral_features(df)
    
    # 4. Divisão Estratificada
    # O uso do 'stratify' é MANDATÓRIO em risco de crédito para manter a taxa de inadimplência 
    # fiel à população original tanto no treino quanto no teste.
    full_df = pd.get_dummies(df.drop(columns=[target], errors='ignore'))
    full_df[target] = df[target]
    
    train, val = train_test_split(
        full_df, test_size=0.2, random_state=42, stratify=full_df[target]
    )

    train.to_csv(abt_dir / cfg["data"]["abt_files"]["abt"], index=False)
    val.to_csv(abt_dir / cfg["data"]["abt_files"]["val"], index=False)
    
    print(f"ABT finalizada. Shape: {full_df.shape}")

if __name__ == "__main__":
    build_abt()