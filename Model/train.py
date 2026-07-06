"""
Módulo de Treinamento e Persistência Dinâmica de Modelos.
"""

from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "Model" / "config.yml"

def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_feature_importance(model, feature_names, save_path, model_name):
    """Gera gráfico de importância e salva dinamicamente."""
    try:
        est = model.named_steps['model'] if isinstance(model, Pipeline) else model
        if hasattr(est, 'feature_importances_'):
            imp = pd.Series(est.feature_importances_, index=feature_names).nlargest(15)
            plt.figure(figsize=(8, 6))
            imp.sort_values().plot(kind='barh', color='skyblue')
            plt.title(f'Top 15 Features: {model_name}')
            plt.tight_layout()
            plt.savefig(save_path / "feature_importance.png")
            plt.close()
    except Exception as e:
        print(f"Feature importance não disponível para {model_name}: {e}")

def build_models(cfg):
    random_state = cfg["project"]["random_state"]
    models_cfg = cfg["models"]

    models = {}

    for model_name, model_info in models_cfg.items():
        if not model_info.get("enabled", True):
            continue

        params = model_info.get("params", {}).copy()
        params["random_state"] = random_state

        if model_info["class"] == "XGBClassifier":
            model = xgb.XGBClassifier(**params)

        elif model_info["class"] == "RandomForestClassifier":
            model = RandomForestClassifier(**params)

        elif model_info["class"] == "LogisticRegression":
            base_model = LogisticRegression(**params)

            if model_info.get("use_scaler", False):
                model = Pipeline([
                    ("scaler", StandardScaler()),
                    ("model", base_model)
                ])
            else:
                model = base_model

        else:
            raise ValueError(f"Modelo não suportado: {model_info['class']}")

        models[model_name] = model

    return models

def train_and_evaluate():
    cfg = load_config()
    target = cfg["project"]["target"]
    abt_dir = PROJECT_ROOT / cfg["paths"]["abt_dir"]
    model_base_dir = PROJECT_ROOT / cfg["paths"]["model_dir"]

    # Carregamento
    train_df = pd.read_csv(abt_dir / "abt_train.csv")
    val_df = pd.read_csv(abt_dir / "val_data.csv")
    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_val, y_val = val_df.drop(columns=[target]), val_df[target]
    
    eval_dir = model_base_dir / "evaluation_data"
    eval_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(X_val, eval_dir / "X_test.pkl")
    joblib.dump(y_val, eval_dir / "y_test.pkl")
    joblib.dump(list(X_val.columns), eval_dir / "feature_names.pkl")

    modelos = build_models(cfg)

    resultados = []

    for nome, model in modelos.items():
        print(f"\n>>> Treinando: {nome}")
        
        # Treino específico
        if nome == 'XGBOOST':
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_train, y_train)

        # Avaliação
        probs = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, probs)
        
        # Threshold
        precision, recall, thresholds = precision_recall_curve(y_val, probs)
        idx = np.where(precision >= 0.35)[0]
        best_thresh = thresholds[idx[0]] if (len(idx) > 0 and idx[0] < len(thresholds)) else 0.5
        resultados.append({'Modelo': nome, 'AUC': auc, 'Threshold': best_thresh})
        
        # PERSISTÊNCIA DINÂMICA
        # O nome do arquivo .pkl agora usa o nome do modelo (ex: XGBOOST.pkl)
        save_path = model_base_dir / nome
        save_path.mkdir(parents=True, exist_ok=True)
        
        save_feature_importance(model, X_train.columns, save_path, nome)
        
        # Salvamento dinâmico: usa a variável 'nome' para criar o nome do arquivo
        joblib.dump(model, save_path / f"{nome.lower()}_model.pkl")
        
        with open(save_path / "threshold.txt", "w") as f:
            f.write(str(best_thresh))
            
    # Rankeamento
    pd.DataFrame(resultados).sort_values(by='AUC', ascending=False).to_csv(model_base_dir / "ranking_modelos.csv", index=False)
    print("\nProcesso finalizado. Modelos salvos dinamicamente em suas pastas.")

if __name__ == "__main__":
    train_and_evaluate()