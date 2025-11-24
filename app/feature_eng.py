import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, ADXIndicator, MACD
from ta.momentum import RSIIndicator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange

def rsrs_features(df: pd.DataFrame, window: int = 18) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula o Indicador RSRS (Regression Slope as a Rating Signal) e R2.
    
    Args:
        df (pd.DataFrame): DataFrame com colunas 'high' e 'low'.
        window (int): Janela de regressão.

    Returns:
        tuple[np.ndarray, np.ndarray]: Arrays NumPy com os valores de Beta (slope) e R2.
    """
    low = df["low"].values
    high = df["high"].values
    betas = np.full(len(df), np.nan, dtype=float)
    r2s = np.full(len(df), np.nan, dtype=float)

    for i in range(window, len(df)):
        x = low[i-window:i].reshape(-1, 1)
        y = high[i-window:i].reshape(-1, 1)
        
        # OLS rápido via closed-form para regressão High = Beta * Low + Alpha
        x_mean = x.mean()
        y_mean = y.mean()
        
        # Cálculo de Covariância e Variância
        cov = ((x - x_mean) * (y - y_mean)).sum()
        var = ((x - x_mean) ** 2).sum()
        
        if var == 0:
            continue
        
        # Beta (Slope)
        beta = cov / var
        betas[i] = beta

        # R2 (Coefficient of Determination)
        alpha = y_mean - beta * x_mean
        y_hat = beta * x + alpha
        ss_res = ((y - y_hat) ** 2).sum()
        ss_tot = ((y - y_mean) ** 2).sum()
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r2s[i] = r2

    return betas, r2s

def adicionar_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adiciona um conjunto robusto de indicadores técnicos e features de risco/momento ao DataFrame.

    Args:
        df (pd.DataFrame): DataFrame com colunas 'open', 'high', 'low', 'close', 'volume'.

    Returns:
        pd.DataFrame: DataFrame com as novas colunas de features, após a remoção de NaNs iniciais.
    """
    df = df.copy()

    # Retornos e ATR (necessário para volatilidade)
    df["ret1"] = np.log(df["close"]).diff(1)
    atr = AverageTrueRange(df["high"], df["low"], df["close"])
    df["atr14"] = atr.average_true_range()

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

    # EMAs
    for w in [9, 20, 21, 50, 200]:
        df[f"ema{w}"] = EMAIndicator(df["close"], w).ema_indicator()

    # Volatilidades
    for w in [5, 10, 20, 60]:
        df[f"vol{w}"] = df["ret1"].rolling(w).std()

    # Z-SCORE (Distância do preço à EMA em relação à volatilidade)
    df["zscore_close_20"] = (df["close"] - df["ema20"]) / df["ret1"].rolling(20).std()

    # Slopes (Momento das médias móveis)
    df["slope_ema21_5"] = df["ema21"].diff(5)
    df["slope_ema50_5"] = df["ema50"].diff(5)

    # RSRS
    beta, r2 = rsrs_features(df, window=18)
    df["rsrs_beta18"] = beta
    df["rsrs_r2_18"] = r2

    # Skew/Kurt (Forma da distribuição de retornos)
    df["skew20"] = df["ret1"].rolling(20).skew()
    df["kurt20"] = df["ret1"].rolling(20).kurt()

    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def criar_target(df: pd.DataFrame, horizonte: int = 5, k: float = 0.4) -> pd.DataFrame:
    """
    Cria a variável target de classificação:
    Se o retorno futuro (Horizonte dias) for maior que k * Volatilidade, a classe é 1 (BUY/ALTA).

    Args:
        df (pd.DataFrame): DataFrame com a coluna 'close' e 'vol20' (volatilidade de 20 dias).
        horizonte (int): Período em dias para calcular o retorno futuro.
        k (float): Multiplicador da volatilidade para definir o limiar de retorno significativo.

    Returns:
        pd.DataFrame: DataFrame com as colunas 'target_ret_h' (retorno) e 'classe' (0 ou 1).
    """
    df = df.copy()
    
    # Retorno futuro (log-retorno)
    # df["close"].shift(-horizonte) garante que o retorno seja do fechamento de hoje até o fechamento futuro
    fut_ret = np.log(df["close"].shift(-horizonte)) - np.log(df["close"])
    df["target_ret_h"] = fut_ret

    # Calcula o limiar de retorno significativo
    # df["vol20"] deve ter sido calculado em adicionar_indicadores
    df = df.dropna(subset=["vol20"])
    thr = k * df["vol20"]
    
    # Classificação (target)
    df["classe"] = (df["target_ret_h"] > thr).astype(int)
    
    df = df.dropna()
    return df
