import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional

def carregar_dados(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Baixa dados históricos de preços de um ticker usando o yfinance.

    Args:
        ticker (str): Símbolo do ativo (e.g., 'MSFT', 'PETR4.SA').
        start_date (Optional[str]): Data inicial para coleta (Formato YYYY-MM-DD).
        end_date (Optional[str]): Data final para coleta (Formato YYYY-MM-DD).

    Returns:
        Optional[pd.DataFrame]: DataFrame com colunas padronizadas ('close', 'open', 'high', 'low', 'volume') 
                                ou None em caso de erro.
    """
    print(f"   → Baixando de {start_date} a {end_date}...")
    try:
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
    except Exception as e:
        print(f"[ERRO] Falha ao baixar dados para {ticker}: {e}")
        return None
        
    if df is None or df.empty:
        print(f"[ERRO] Sem dados para {ticker} no período especificado.")
        return None

    # Normalização de nomes de colunas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(c).lower().strip().replace(" ", "_") for c in col if c])
            for col in df.columns
        ]
    else:
        df.columns = [str(c).lower().strip().replace(" ", "_") for c in df.columns]

    # Mapeamento flexível das colunas necessárias
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

    # Tratamento para colunas faltantes com fallback SEM vazamento de dados futuros
    if colmap["open"] is not None:
        out["open"] = df[colmap["open"]]
    else:
        out["open"] = out["close"].shift(1) # Usa o fechamento anterior

    if colmap["high"] is not None:
        out["high"] = df[colmap["high"]]
    else:
        out["high"] = out[["open", "close"]].max(axis=1) # Max entre open e close

    if colmap["low"] is not None:
        out["low"] = df[colmap["low"]]
    else:
        out["low"] = out[["open", "close"]].min(axis=1) # Min entre open e close

    if colmap["volume"] is not None:
        out["volume"] = df[colmap["volume"]]
    else:
        out["volume"] = 0

    out = out.dropna()
    return out
