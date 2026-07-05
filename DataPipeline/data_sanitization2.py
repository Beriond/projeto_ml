"""Sanitização dos dados brutos (raw -> clean).

Todos os parâmetros (caminhos, nomes de arquivo, estratégias de imputação,
limites de nulos, valor alvo) vêm do config.yml — nada é chumbado aqui.
Os caminhos são resolvidos relativos à raiz do projeto via pathlib, de forma
portável entre Linux, macOS e Windows. Saídas gravadas direto em /Dados (flat),
seguindo a estrutura de entrega.
"""

from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "DataPipeline" / "config.yml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_application(df: pd.DataFrame, cfg: dict, is_train: bool) -> pd.DataFrame:
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, pd.NA)
    df["IDADE_ANOS"] = abs(df["DAYS_BIRTH"]) / 365

    if is_train and "TARGET" in df.columns:
        if not set(df["TARGET"].dropna()).issubset({0, 1}):
            raise ValueError("TARGET contém valores inválidos.")
    return df


def sanitize_previous_application(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_fix = ['DAYS_FIRST_DRAWING', 'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION',
                   'DAYS_LAST_DUE', 'DAYS_TERMINATION']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(365243, pd.NA)
    if 'AMT_APPLICATION' in df.columns:
        df['AMT_APPLICATION'] = df['AMT_APPLICATION'].clip(lower=0)
    return df


def sanitize_bureau(df: pd.DataFrame) -> pd.DataFrame:
    """Limpeza básica para a base Bureau."""
    # Tratar valores  comuns em datas do Bureau
    cols_to_fix = ['DAYS_CREDIT', 'DAYS_CREDIT_ENDDATE', 'DAYS_ENDDATE_FACT']
    for col in cols_to_fix:
        if col in df.columns:
            df[col] = df[col].replace(365243, pd.NA)
    return df


def run_sanitization(cfg: dict | None = None) -> None:
    cfg = cfg or load_config()
    raw_dir = PROJECT_ROOT / cfg["paths"]["raw_dir"]
    clean_dir = PROJECT_ROOT / cfg["paths"]["clean_dir"]
    clean_dir.mkdir(parents=True, exist_ok=True)

    # Função de limpeza diretamente na tupla (nome_raw, nome_clean, é_treino, funcao_limpeza)
    tasks = [
        (cfg["data"]["raw_files"]["application"], cfg["data"]["clean_files"]["application"], True, sanitize_application),
        (cfg["data"]["raw_files"]["previous_application"], cfg["data"]["clean_files"]["previous_application"], False, sanitize_previous_application),
        (cfg["data"]["raw_files"]["bureau"], cfg["data"]["clean_files"]["bureau"], False, sanitize_bureau)
    ]

    for raw_name, clean_name, is_train, sanitize_func in tasks:
        input_path = raw_dir / raw_name
        if input_path.exists():
            print(f"--- Sanitizando: {raw_name} ---")
            df = pd.read_csv(input_path)

            # Chama a função específica definida na tupla
            if sanitize_func == sanitize_application:
                df_clean = sanitize_func(df, cfg, is_train)
            else:
                df_clean = sanitize_func(df)

            df_clean = df_clean.drop_duplicates()
            output_path = clean_dir / clean_name
            df_clean.to_csv(output_path, index=False)
            print(f"Salvo em: {output_path}")


if __name__ == "__main__":
    run_sanitization()
