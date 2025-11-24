import pandas as pd
import numpy as np
from typing import Dict, Any

def backtest_simples(df: pd.DataFrame, oof_pred: pd.Series, limiar: float = 0.60, horizonte: int = 5) -> Dict[str, Any]:
    """
    Simulação de backtest simples (compra e hold por H dias).
    - Sinal: Probabilidade OOF > Limiar.
    - Entrada: No fechamento do dia do sinal.
    - Saída: No fechamento H dias depois.

    Args:
        df (pd.DataFrame): DataFrame com a coluna 'close'.
        oof_pred (pd.Series): Previsões OOF (Out-of-Fold) geradas pelo modelo.
        limiar (float): Limiar mínimo da probabilidade para gerar um sinal BUY.
        horizonte (int): Período de hold em dias.

    Returns:
        Dict[str, Any]: Resultados do backtest.
    """
    df = df.copy()

    # Garantir alinhamento temporal e usar apenas previsões válidas
    df["proba"] = oof_pred
    df = df.dropna(subset=["proba"])

    sinais = df["proba"] > limiar

    entradas = df.index[sinais]
    resultados = []

    for entrada in entradas:
        idx = df.index.get_loc(entrada)

        # Checar se existe saída válida dentro do horizonte
        if idx + horizonte >= len(df):
            continue

        preco_entrada = df["close"].iloc[idx]
        preco_saida   = df["close"].iloc[idx + horizonte]

        retorno = np.log(preco_saida) - np.log(preco_entrada)

        resultados.append(retorno)

    if not resultados:
        return {
            "trades": 0, "ret_total": 0.0, "ret_medio": 0.0,
            "taxa_acerto": 0.0, "sharpe": 0.0
        }

    retornos = np.array(resultados)
    
    r_std = retornos.std()
    # Sharpe Ratio: Média / Desvio Padrão
    sharpe = retornos.mean() / (r_std + 1e-9) if r_std != 0 else 0
    
    return {
        "trades": int(len(retornos)),
        "ret_total": float(retornos.sum()),
        "ret_medio": float(retornos.mean()),
        "taxa_acerto": float((retornos > 0).mean()),
        "sharpe": float(sharpe)
    }

def backtest_avancado(df: pd.DataFrame, oof_pred: pd.Series, limiar: float = 0.60, horizonte: int = 5, atr_mult: float = 1.5) -> Dict[str, Any]:
    """
    Simulação de backtest avançado com filtro de tendência (EMA) e Stop Loss dinâmico (ATR).
    - Sinal: Probabilidade OOF > Limiar E EMA(21) > EMA(50).
    - Entrada: Na abertura (Open) do dia seguinte ao sinal (D+1).
    - Stop Loss: (Preço de Entrada - ATR * ATR_Mult).
    - Saída: No Stop, ou no fechamento do dia (D+1+horizonte).

    Args:
        df (pd.DataFrame): DataFrame com colunas OHLC, 'atr14', 'ema21', 'ema50'.
        oof_pred (pd.Series): Previsões OOF (Out-of-Fold) geradas pelo modelo.
        limiar (float): Limiar mínimo da probabilidade para gerar um sinal BUY.
        horizonte (int): Período máximo de hold.
        atr_mult (float): Multiplicador do ATR para definir o Stop Loss.

    Returns:
        Dict[str, Any]: Resultados do backtest.
    """
    df = df.copy()
    df["proba"] = oof_pred
    
    # Verifica se todas as colunas necessárias estão presentes
    req_cols = {"open","high","low","close","ema21","ema50","atr14","proba"}
    if not req_cols.issubset(df.columns):
        return {"trades":0,"ret_total":0.0,"ret_medio":0.0,"taxa_acerto":0.0,"sharpe":0.0}

    # Condição de Sinal (Probabilidade ALTA + Filtro de Tendência)
    sinais = (df["proba"] > limiar) & (df["ema21"] > df["ema50"])
    entradas_idx = df.index[sinais]

    trades_returns = []
    # exit_until guarda o último dia do trade anterior. Só pode entrar no dia seguinte.
    exit_until = df.index[0] 

    for d in entradas_idx:
        i = df.index.get_loc(d)
        
        # Ignora se o sinal cai dentro de um trade em andamento
        if i <= df.index.get_loc(exit_until):
            continue

        # 1. ENTRADA (Sempre D+1 na abertura)
        if i + 1 >= len(df): continue
        entry_day = df.index[i + 1]
        entry_open = df.loc[entry_day, "open"]
        if pd.isna(entry_open): continue

        # 2. STOP LOSS (Calculado com ATR do DIA DO SINAL (D))
        atr = df.loc[d, "atr14"]
        if pd.isna(atr) or atr <= 0: continue
        stop_level = entry_open - atr_mult * atr

        # 3. JANELA DE GESTÃO E SAÍDA
        max_exit_i = min(i + 1 + horizonte, len(df) - 1)
        realized = False
        ret = 0.0

        # Procura por Stop (a partir do dia D+1 até o max_exit_i)
        for j in range(i + 1, max_exit_i + 1):
            day = df.index[j]
            day_low = df.loc[day, "low"]
            
            # Se o low bater abaixo do stop, saída no STOP no mesmo dia
            if day_low <= stop_level:
                exit_price = stop_level
                ret = np.log(exit_price) - np.log(entry_open)
                realized = True
                exit_until = day # Marca a saída
                break

        # Se não acionou stop, sai no fechamento do último dia da janela
        if not realized:
            last_day = df.index[max_exit_i]
            exit_price = df.loc[last_day, "close"]
            ret = np.log(exit_price) - np.log(entry_open)
            exit_until = last_day # Marca a saída
            
        trades_returns.append(ret)
        
        # Atualiza o índice de saída para evitar trades sobrepostos
        # exit_until é o dia em que o trade foi encerrado. O próximo trade deve começar em exit_until + 1.
        if df.index.get_loc(exit_until) < len(df) - 1:
            continue
        else:
             break # Se a saída for no último dia, termina o loop

    if not trades_returns:
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
