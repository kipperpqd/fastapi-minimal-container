import pandas as pd
import numpy as np
import yfinance as yf
import warnings
import json
from datetime import datetime, date
from typing import Optional, List, Dict, Any

# FastAPI e Pydantic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Funções de Análise (Importações do seu código)
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score

warnings.filterwarnings("ignore")

# CatBoost (opcional – fallback se não estiver instalado)
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except Exception:
    HAS_CATBOOST = False

# ===============================================
# CONFIGURAÇÃO DO FASTAPI E MODELOS PYDANTIC
# ===============================================
app = FastAPI(title="Quant Trading Signal API", version="1.0.0")

# Input Model (Define o JSON que o usuário irá enviar)
class SignalRequest(BaseModel):
    """Modelo de entrada para o endpoint de análise de sinais."""
    tickers: List[str] = Field(..., description="Lista de símbolos de ativos (e.g., MSFT, PETR4.SA, BTC-USD).")
    start_date: Optional[str] = Field("2010-01-01", description="Data inicial para coleta de dados (Formato YYYY-MM-DD).")
    end_date: Optional[str] = Field(None, description="Data final para coleta de dados (Formato YYYY-MM-DD). Se nulo, usa a data atual.")
    horizonte_dias: int = Field(5, description="Horizonte de previsão do target (dias).")
    k_volatilidade: float = Field(0.4, description="Multiplicador de volatilidade para definir o target de retorno significativo.")
    entry_threshold: float = Field(0.60, description="Limiar mínimo de probabilidade para gerar um sinal BUY.")
    atr_multiplier: float = Field(1.5, description="Multiplicador do ATR para o Stop Loss no Backtest Avançado.")

# Output Model (Define o JSON que a API irá retornar)
class BacktestResult(BaseModel):
    trades: int
    ret_total: float
    ret_medio: float
    taxa_acerto: float
    sharpe: float

class SignalResult(BaseModel):
    """Modelo de saída para o resultado da análise de um ativo."""
    ticker: str
    probabilidade_alta: float
    sinal_binario: str
    AUC: Optional[float]
    ACC: Optional[float]
    F1: Optional[float]
    AP: Optional[float]
    backtest_simples: BacktestResult
    backtest_avancado: BacktestResult
    status: str

# ===============================================
# LÓGICA DE TRADING (Funções da sua implementação)
# ===============================================

# 2) Baixar dados (mantido do seu código)
def carregar_dados(ticker, start_date=None, end_date=None):
    """
    Baixa dados históricos do yfinance usando start/end date, em vez de period="10y".
    Usa um intervalo de 1 dia ('1d').
    """
    # ... (Seu código original da função carregar_dados) ...
    print(f"   → Baixando de {start_date} a {end_date}...")
    df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    
    if df is None or df.empty:
        print(f"[ERRO] Sem dados para {ticker} no período especificado.")
        return None

    # Flatten de MultiIndex e normalização de nomes
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c).lower().strip().replace(" ", "_") for c in col if c])
            for col in df.columns
        ]
    else:
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # Mapeamento flexível por substrings
    colmap = {"open": None, "high": None, "low": None, "close": None, "volume": None}
    for c in df.columns:
        if ("close" in c) or ("adj" in c and "close" in c):
            colmap["close"] = colmap["close"] or c
        if "open" in c:
            colmap["open"] = colmap["open"] or c
        if "high" in c:
            colmap["high"] = colmap["high"] or c
        if "low" in c:
            colmap["low"] = colmap["low"] or c
        if "vol" in c:
            colmap["volume"] = colmap["volume"] or c

    if colmap["close"] is None:
        print(f"[ERRO] {ticker}: sem coluna CLOSE → pulando.")
        return None

    out = pd.DataFrame(index=df.index)
    out["close"] = df[colmap["close"]]

    # Open/High/Low/Volume com fallback SEM vazamento
    if colmap["open"] is not None:
        out["open"] = df[colmap["open"]]
    else:
        out["open"] = out["close"].shift(1)

    if colmap["high"] is not None:
        out["high"] = df[colmap["high"]]
    else:
        out["high"] = out[["open", "close"]].max(axis=1)

    if colmap["low"] is not None:
        out["low"] = df[colmap["low"]]
    else:
        out["low"] = out[["open", "close"]].min(axis=1)

    if colmap["volume"] is not None:
        out["volume"] = df[colmap["volume"]]
    else:
        out["volume"] = 0

    out = out.dropna()
    return out

# 3) Indicadores técnicos + features avançadas (mantido do seu código)
def rsrs_features(df, window=18):
    # ... (Seu código original da função rsrs_features) ...
    low = df["low"].values
    high = df["high"].values
    betas = np.full(len(df), np.nan, dtype=float)
    r2s = np.full(len(df), np.nan, dtype=float)

    for i in range(window, len(df)):
        x = low[i-window:i].reshape(-1, 1)
        y = high[i-window:i].reshape(-1, 1)
        # OLS rápido via closed-form
        x_mean = x.mean()
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        if var == 0:
            continue
        beta = cov / var
        y_hat = beta * (x - x_mean) + y_mean
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        betas[i] = beta
        r2s[i] = r2

    return betas, r2s

def adicionar_indicadores(df):
    # ... (Seu código original da função adicionar_indicadores) ...
    df = df.copy()

    # Indicadores clássicos    
    df["rsi14"] = RSIIndicator(df["close"], 14).rsi()

    macd = MACD(df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()

    bb = BollingerBands(df["close"])
    df["bb_u"] = bb.bollinger_hband()
    df["bb_l"] = bb.bollinger_lband()
    df["bb_pct"] = (df["close"] - df["bb_l"]) / (df["bb_u"] - df["bb_l"])

    df["adx14"] = ADXIndicator(df["high"], df["low"], df["close"]).adx()
    df["roc5"] = ROCIndicator(df["close"], 5).roc()

    # EMA's
    for w in [9, 20, 21, 50, 200]:
        df[f"ema{w}"] = EMAIndicator(df["close"], w).ema_indicator()

    atr = AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr14"] = atr.average_true_range()

    # Retornos e volatilidades
    df["ret1"] = np.log(df["close"]).diff(1)

    for w in [5, 10, 20, 60]:
        df[f"vol{w}"] = df["ret1"].rolling(w).std()

    # Z-SCORE
    df["zscore_close_20"] = (df["close"] - df["ema20"]) / df["ret1"].rolling(20).std()

    # Slopes
    df["slope_ema21_5"] = df["ema21"].diff(5)
    df["slope_ema50_5"] = df["ema50"].diff(5)

    # RSRS
    beta, r2 = rsrs_features(df, window=18)
    df["rsrs_beta18"] = beta
    df["rsrs_r2_18"] = r2

    # Skew/Kurt
    df["skew20"] = df["ret1"].rolling(20).skew()
    df["kurt20"] = df["ret1"].rolling(20).kurt()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

# 4) Target inteligente (mantido do seu código)
def criar_target(df, horizonte=5, k=0.4):
    # ... (Seu código original da função criar_target) ...
    df = df.copy()
    # O retorno futuro é calculado usando o preço de fechamento no dia (i + horizonte)
    fut_ret = np.log(df["close"].shift(-horizonte)) - np.log(df["close"])
    df["target_ret_h"] = fut_ret
    # vol20 já calculada em adicionar_indicadores
    df = df.dropna(subset=["vol20"])
    thr = k * df["vol20"]
    df["classe"] = (df["target_ret_h"] > thr).astype(int)
    df = df.dropna()
    return df

# 5) Treino Walk-Forward (mantido do seu código)
def treinar_modelo(df, features, n_splits=5, entry_thr=0.55, seed=42):
    # ... (Seu código original da função treinar_modelo) ...
    X = df[features]
    y = df["classe"]

    tscv = TimeSeriesSplit(n_splits=n_splits)

    oof_xgb = pd.Series(index=y.index, dtype=float)
    oof_cat = pd.Series(index=y.index, dtype=float) if HAS_CATBOOST else None

    # Inicializa os modelos
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

    # Métricas com máscaras válidas
    mask_valid = oof_ens.notna()
    if mask_valid.sum() < 10:
        print("[AVISO] Pouca previsão válida neste ativo → pulando métricas.")
        modelos = {"xgb": xgb_clf}
        if HAS_CATBOOST:
            modelos["cat"] = cat_clf
        return modelos, scaler, oof_ens, {"AUC": None, "ACC": None, "F1": None, "AP": None}

    yv = y[mask_valid]
    pv = oof_ens[mask_valid]
    auc = roc_auc_score(yv, pv)
    pred_cls = (pv > entry_thr).astype(int)
    acc = accuracy_score(yv, pred_cls)
    f1 = f1_score(yv, pred_cls)
    ap = average_precision_score(yv, pv)

    modelos = {"xgb": xgb_clf}
    if HAS_CATBOOST:
        modelos["cat"] = cat_clf

    return modelos, scaler, oof_ens, {"AUC": auc, "ACC": acc, "F1": f1, "AP": ap}

# 6) Prever o último dia (mantido do seu código)
def prever_ultimo_dia(df, modelos, scaler, features):
    # ... (Seu código original da função prever_ultimo_dia) ...
    X_last = df[features].iloc[-1:]
    X_last_sc = scaler.transform(X_last)

    p = 0.0
    p += modelos["xgb"].predict_proba(X_last_sc)[0][1]
    if "cat" in modelos:
        p += modelos["cat"].predict_proba(X_last_sc)[0][1]
        p /= 2.0
    return float(p)

# 8) Backtest simples (mantido do seu código)
def backtest_simples(df, oof_pred, limiar=0.60, horizonte=5):
    # ... (Seu código original da função backtest_simples) ...
    df = df.copy()

    # Garantir alinhamento temporal (muito importante)
    df["proba"] = oof_pred
    df = df.dropna(subset=["proba"])

    sinais = df["proba"] > limiar

    entradas = df.index[sinais]

    resultados = []

    for entrada in entradas:
        idx = df.index.get_loc(entrada)

        # Checar se existe saída válida
        if idx + horizonte >= len(df):
            continue

        preco_entrada = df["close"].iloc[idx]
        preco_saida   = df["close"].iloc[idx + horizonte]

        retorno = np.log(preco_saida) - np.log(preco_entrada)

        resultados.append(retorno)

    if len(resultados) == 0:
        return {
            "trades": 0,
            "ret_total": 0.0,
            "ret_medio": 0.0,
            "taxa_acerto": 0.0,
            "sharpe": 0.0
        }

    retornos = np.array(resultados)
    
    # Prevenção contra STD zero
    r_std = retornos.std()
    sharpe = retornos.mean() / (r_std + 1e-9) if r_std != 0 else 0
    
    return {
        "trades": int(len(retornos)),
        "ret_total": float(retornos.sum()),
        "ret_medio": float(retornos.mean()),
        "taxa_acerto": float((retornos > 0).mean()),
        "sharpe": float(sharpe)
    }

# 9) Backtest avançado (mantido do seu código)
def backtest_avancado(df, oof_pred, limiar=0.60, horizonte=5, atr_mult=1.5):
    # ... (Seu código original da função backtest_avancado) ...
    df = df.copy()
    df["proba"] = oof_pred
    # Precisamos desses campos calculados previamente:
    req_cols = {"open","high","low","close","ema21","ema50","atr14","proba"}
    # Verifica se todas as colunas necessárias estão presentes
    if not req_cols.issubset(df.columns):
        return {"trades":0,"ret_total":0.0,"ret_medio":0.0,"taxa_acerto":0.0,"sharpe":0.0}

    # Sinal no dia D (usado só para decidir), entrada em D+1 open
    sinais = (df["proba"] > limiar) & (df["ema21"] > df["ema50"])
    entradas_idx = df.index[sinais]

    trades_returns = []
    in_trade = False
    exit_until = df.index[0] # Inicializa com um índice seguro

    for d in entradas_idx:
        i = df.index.get_loc(d)
        
        # Se in_trade for True, checa se o sinal está dentro do trade em andamento
        if in_trade and i <= df.index.get_loc(exit_until):
            continue

        # Entrada sempre D+1 na abertura
        if i + 1 >= len(df):
            continue
        
        entry_day = df.index[i + 1]
        entry_open = df.loc[entry_day, "open"]
        if pd.isna(entry_open):
            continue

        # Stop calculado com ATR do DIA DO SINAL (D)
        atr = df.loc[d, "atr14"]
        if pd.isna(atr) or atr <= 0:
            continue
        stop_level = entry_open - atr_mult * atr

        # Janela de gestão: do D+1 (entrada) até D+1+horizonte (saída máxima)
        max_exit_i = min(i + 1 + horizonte, len(df) - 1)
        realized = False
        ret = 0.0

        # Percorre cada dia da janela procurando stop (a partir do dia D+1)
        for j in range(i + 1, max_exit_i + 1):
            day = df.index[j]
            day_low = df.loc[day, "low"]
            
            # Se o low bater abaixo do stop, saída no STOP no mesmo dia
            if day_low <= stop_level:
                exit_price = stop_level
                ret = np.log(exit_price) - np.log(entry_open)
                realized = True
                exit_until = day
                break

        # Se não acionou stop, sai no fechamento do último dia da janela
        if not realized:
            last_day = df.index[max_exit_i]
            exit_price = df.loc[last_day, "close"]
            ret = np.log(exit_price) - np.log(entry_open)
            exit_until = last_day
            
        trades_returns.append(ret)
        
        # Atualiza a marcação de que um trade está em andamento (até exit_until)
        # O próximo trade só pode entrar no dia após exit_until
        if df.index.get_loc(exit_until) < len(df) - 1:
            in_trade = True 
        else:
             in_trade = False

    if len(trades_returns) == 0:
        return {"trades":0,"ret_total":0.0,"ret_medio":0.0,"taxa_acerto":0.0,"sharpe":0.0}

    r = np.array(trades_returns)
    r_std = r.std()
    sharpe = r.mean() / (r_std + 1e-9) if r_std != 0 else 0
    
    return {
        "trades": int(len(r)),
        "ret_total": float(r.sum()),
        "ret_medio": float(r.mean()),
        "taxa_acerto": float((r > 0).mean()),
        "sharpe": float(sharpe)
    }

# 7) Pipeline principal (Modificado para receber a lista de tickers)
def processar_ativos_api(
    tickers: List[str],
    start_date: str,
    end_date: Optional[str] = None,
    horizonte: int = 5,
    k: float = 0.4,
    entry_thr: float = 0.60,
    atr_mult: float = 1.5
) -> List[Dict[str, Any]]:
    """Executa o pipeline completo de análise para uma lista de ativos."""
    if not tickers:
        return []

    if end_date is None:
        # Se end_date for None, usa o dia anterior ao atual (para yfinance)
        end_date = date.today().strftime("%Y-%m-%d")

    resultados_lista = []

    features = [
        # features... (todas as suas features)
        "rsi14","macd","macd_signal","macd_hist",
        "bb_pct","adx14","roc5",
        "ema9","ema21","ema50","ema200",
        "atr14","ret1",
        "vol5","vol10","vol20","vol60",
        "zscore_close_20",
        "slope_ema21_5","slope_ema50_5",
        "rsrs_beta18","rsrs_r2_18",
        "skew20","kurt20"
    ]

    for ticker in tickers:
        print(f"\n🔍 Processando {ticker}...")
        
        try:
            df = carregar_dados(ticker, start_date=start_date, end_date=end_date)
            
            if df is None:
                resultados_lista.append({"ticker": ticker, "status": "Erro: Dados não encontrados ou insuficientes."})
                continue

            df = adicionar_indicadores(df)
            df = criar_target(df, horizonte=horizonte, k=k)

            # checagem de cobertura de features
            missing = [c for c in features if c not in df.columns]
            if missing:
                print(f"[AVISO] {ticker}: faltam features {missing} → pulando ativo.")
                resultados_lista.append({"ticker": ticker, "status": "Aviso: Faltam features calculadas."})
                continue

            if df.shape[0] < 500:
                print(f"[AVISO] Poucos dados em {ticker}. Pulando.")
                resultados_lista.append({"ticker": ticker, "status": f"Aviso: Poucos dados ({df.shape[0]} linhas) para treinamento robusto."})
                continue

            modelos, scaler, oof_pred, metrics = treinar_modelo(df, features, n_splits=5, entry_thr=entry_thr)

            if metrics["AUC"] is None:
                resultados_lista.append({"ticker": ticker, "status": "Aviso: Métricas inválidas (poucas previsões válidas)." })
                continue

            prob = prever_ultimo_dia(df, modelos, scaler, features)
            sinal_binario = "BUY" if prob > entry_thr else "HOLD/SELL"

            # Backtests
            bt = backtest_simples(df, oof_pred, limiar=entry_thr, horizonte=horizonte)
            bt_adv = backtest_avancado(df, oof_pred, limiar=entry_thr, horizonte=horizonte, atr_mult=atr_mult)

            resultados_lista.append(SignalResult(
                ticker=ticker,
                probabilidade_alta=round(prob, 4),
                sinal_binario=sinal_binario,
                AUC=round(metrics["AUC"], 4),
                ACC=round(metrics["ACC"], 4),
                F1=round(metrics["F1"], 4),
                AP=round(metrics["AP"], 4),
                backtest_simples=BacktestResult(**bt),
                backtest_avancado=BacktestResult(**bt_adv),
                status="Sucesso"
            ).model_dump()) # Usar model_dump() para converter Pydantic em dict

        except Exception as e:
            print(f"  ❌ Erro catastrófico ao processar {ticker}: {e}")
            resultados_lista.append({"ticker": ticker, "status": f"Erro interno: {type(e).__name__}"})


    return resultados_lista

# ===============================================
# ENDPOINTS DA API
# ===============================================

@app.get("/")
async def root():
    """Endpoint de saúde para verificar se a API está online."""
    return {"status": "online", "service": "Quant Trading Analysis API"}

@app.post("/analyze_signals", response_model=List[SignalResult])
async def analyze_signals(request: SignalRequest):
    """
    Executa a análise completa de Machine Learning (Dados -> Features -> Treino Walk-Forward -> Sinal) 
    para a lista de ativos fornecida.

    Retorna a probabilidade de alta e métricas de backtest para cada ativo.
    """
    print(f"Requisição recebida para {len(request.tickers)} ativos.")
    
    # Chama a função principal com os parâmetros da requisição
    results = processar_ativos_api(
        tickers=request.tickers,
        start_date=request.start_date,
        end_date=request.end_date,
        horizonte=request.horizonte_dias,
        k=request.k_volatilidade,
        entry_thr=request.entry_threshold,
        atr_mult=request.atr_multiplier
    )

    # Verifica se todos os resultados foram erros ou avisos
    if not any(isinstance(r, dict) and r.get('status') == 'Sucesso' for r in results):
        # Retorna o status 200, mas com uma mensagem de alerta no corpo
        return results

    # Retorna a lista de resultados (FastAPI serializa automaticamente para JSON)
    return results

@app.get("/metrics")
async def get_metrics_info():
    """Descreve as métricas retornadas pela API."""
    return {
        "metrics_model": {
            "AUC": "Area Under ROC Curve (mede a separação de classes)",
            "ACC": "Acurácia (taxa de acerto da classificação)",
            "F1": "Média harmônica de precisão e recall (bom para classes desbalanceadas)",
            "AP": "Average Precision (média das precisões obtidas em diferentes thresholds)",
        },
        "backtest_fields": {
            "trades": "Número de trades simulados",
            "ret_total": "Retorno logarítmico total",
            "sharpe": "Relação Retorno/Risco (quanto maior, melhor)"
        }
    }
