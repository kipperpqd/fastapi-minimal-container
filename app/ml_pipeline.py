import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score

warnings.filterwarnings("ignore")

# CatBoost (opcional – necessário para o ensemble)
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

def treinar_modelo(
    df: pd.DataFrame, 
    features: list[str], 
    n_splits: int = 5, 
    entry_thr: float = 0.55, 
    seed: int = 42
) -> Tuple[Dict[str, Any], StandardScaler, pd.Series, Dict[str, Optional[float]]]:
    """
    Treina modelos usando TimeSeriesSplit (Walk-Forward) para gerar previsões Out-of-Fold (OOF).

    Args:
        df (pd.DataFrame): DataFrame com features e a coluna 'classe' (target).
        features (list[str]): Lista de colunas a serem usadas como features.
        n_splits (int): Número de splits para o TimeSeriesSplit.
        entry_thr (float): Limiar para calcular métricas de classificação binária (ACC, F1).
        seed (int): Semente aleatória para reprodutibilidade.

    Returns:
        Tuple[Dict, StandardScaler, pd.Series, Dict]: Modelos treinados, Scaler, OOF predictions e Métricas.
    """
    X = df[features]
    y = df["classe"]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_xgb = pd.Series(index=y.index, dtype=float)
    oof_cat = pd.Series(index=y.index, dtype=float) if HAS_CATBOOST else None

    # Inicializa os modelos de base
    xgb_clf = xgb.XGBClassifier(
        n_estimators=600, max_depth=5, learning_rate=0.03, subsample=0.85,
        colsample_bytree=0.85, reg_lambda=1.2, reg_alpha=0.0, min_child_weight=3,
        objective="binary:logistic", eval_metric="logloss", random_state=seed, use_label_encoder=False
    )
    if HAS_CATBOOST:
        cat_clf = CatBoostClassifier(
            depth=6, learning_rate=0.03, iterations=700, l2_leaf_reg=3.0,
            subsample=0.85, loss_function="Logloss", verbose=False, random_state=seed
        )

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Escalonador sem vazamento (apenas dados de treino)
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc  = scaler.transform(X_test)

        # ---- XGBoost
        xgb_clf.fit(X_train_sc, y_train)
        oof_xgb.iloc[test_idx] = xgb_clf.predict_proba(X_test_sc)[:, 1]

        # ---- CatBoost (se disponível)
        if HAS_CATBOOST:
            cat_clf.fit(X_train_sc, y_train)
            oof_cat.iloc[test_idx] = cat_clf.predict_proba(X_test_sc)[:, 1]

    # Ensemble OOF
    if HAS_CATBOOST:
        oof_ens = 0.5 * oof_xgb + 0.5 * oof_cat
    else:
        oof_ens = oof_xgb.copy()

    # Cálculo das Métricas
    mask_valid = oof_ens.notna()
    if mask_valid.sum() < 10:
        print("[AVISO] Pouca previsão válida neste ativo → pulando métricas.")
        modelos = {"xgb": xgb_clf}
        if HAS_CATBOOST: modelos["cat"] = cat_clf
        return modelos, scaler, oof_ens, {"AUC": None, "ACC": None, "F1": None, "AP": None}

    yv = y[mask_valid]
    pv = oof_ens[mask_valid]
    
    auc = roc_auc_score(yv, pv)
    pred_cls = (pv > entry_thr).astype(int)
    acc = accuracy_score(yv, pred_cls)
    f1 = f1_score(yv, pred_cls)
    ap = average_precision_score(yv, pv)

    modelos = {"xgb": xgb_clf}
    if HAS_CATBOOST: modelos["cat"] = cat_clf

    return modelos, scaler, oof_ens, {"AUC": auc, "ACC": acc, "F1": f1, "AP": ap}

def prever_ultimo_dia(df: pd.DataFrame, modelos: Dict[str, Any], scaler: StandardScaler, features: list[str]) -> float:
    """
    Usa o modelo treinado e o scaler para prever a probabilidade de alta para o último dia disponível.

    Args:
        df (pd.DataFrame): DataFrame com as features calculadas.
        modelos (Dict[str, Any]): Dicionário com os modelos treinados ('xgb', 'cat').
        scaler (StandardScaler): O scaler treinado no processo OOF.
        features (list[str]): Lista de colunas a serem usadas como features.

    Returns:
        float: Probabilidade de alta (média do ensemble).
    """
    X_last = df[features].iloc[-1:]
    X_last_sc = scaler.transform(X_last)

    p = 0.0
    p += modelos["xgb"].predict_proba(X_last_sc)[0][1]
    
    if "cat" in modelos:
        p += modelos["cat"].predict_proba(X_last_sc)[0][1]
        p /= 2.0
        
    return float(p)
