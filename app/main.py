import pandas as pd
import numpy as np
import warnings
from datetime import date
from typing import Optional, List, Dict, Any

# FastAPI e Pydantic (Usado para definir a estrutura dos dados de entrada/saída da API)
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# ====================================================================
# IMPORTAÇÕES DOS NOVOS MÓDULOS (A orquestração chama as funções de cada arquivo)
# Certifique-se de que estes arquivos existam na pasta 'app/'
# ====================================================================
from .data_loader import carregar_dados
from .feature_eng import adicionar_indicadores, criar_target
from .ml_pipeline import treinar_modelo, prever_ultimo_dia
from .backtester import backtest_simples, backtest_avancado

warnings.filterwarnings("ignore")

# ===============================================
# CONFIGURAÇÃO DO FASTAPI E MODELOS PYDANTIC
# ===============================================
# Removemos o endpoint /ping e os endpoints de teste (webhook, clientes)
app = FastAPI(title="Quant Trading Signal API", version="1.0.1")

# Input Model (Define o JSON que o usuário irá enviar para /analyze_signals)
class SignalRequest(BaseModel):
    """Modelo de entrada para o endpoint de análise de sinais."""
    tickers: List[str] = Field(..., description="Lista de símbolos de ativos (e.g., MSFT, PETR4.SA, BTC-USD).")
    start_date: Optional[str] = Field("2010-01-01", description="Data inicial para coleta de dados (Formato YYYY-MM-DD).")
    end_date: Optional[str] = Field(None, description="Data final para coleta de dados (Formato YYYY-MM-DD). Se nulo, usa a data atual.")
    horizonte_dias: int = Field(5, description="Horizonte de previsão do target (dias).")
    k_volatilidade: float = Field(0.4, description="Multiplicador de volatilidade para definir o target de retorno significativo.")
    entry_threshold: float = Field(0.60, description="Limiar mínimo de probabilidade para gerar um sinal BUY.")
    atr_multiplier: float = Field(1.5, description="Multiplicador do ATR para o Stop Loss no Backtest Avançado.")

# Output Models (Define a estrutura da resposta JSON)
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
    # Tornamos opcional pois um erro no pipeline pode impedir o cálculo
    backtest_simples: Optional[BacktestResult] 
    backtest_avancado: Optional[BacktestResult]
    status: str

# ===============================================
# LÓGICA DE ORQUESTRAÇÃO (Função que coordena os módulos)
# ===============================================
# Esta função é a única que manteve a lógica do seu pipeline, mas agora chama as funções externas.
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
        end_date = date.today().strftime("%Y-%m-%d")

    resultados_lista = []

    # A lista de features permanece aqui, pois a orquestração precisa saber o que esperar.
    features = [
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
            # 1. Carregar Dados (Chama data_loader.py)
            df = carregar_dados(ticker, start_date=start_date, end_date=end_date)
            
            if df is None:
                resultados_lista.append({"ticker": ticker, "status": "Erro: Dados não encontrados ou insuficientes."})
                continue

            # 2. Adicionar Indicadores e Features (Chama feature_eng.py)
            df = adicionar_indicadores(df)
            df = criar_target(df, horizonte=horizonte, k=k)

            # Validação e avisos... (Mantido do seu código)
            missing = [c for c in features if c not in df.columns]
            if missing:
                print(f"[AVISO] {ticker}: faltam features {missing} → pulando ativo.")
                resultados_lista.append({"ticker": ticker, "status": "Aviso: Faltam features calculadas."})
                continue

            if df.shape[0] < 500:
                print(f"[AVISO] Poucos dados em {ticker}. Pulando.")
                resultados_lista.append({"ticker": ticker, "status": f"Aviso: Poucos dados ({df.shape[0]} linhas) para treinamento robusto."})
                continue
            
            # 3. Treinar Modelo (Chama ml_pipeline.py)
            modelos, scaler, oof_pred, metrics = treinar_modelo(df, features, n_splits=5, entry_thr=entry_thr)

            if metrics["AUC"] is None:
                resultados_lista.append({"ticker": ticker, "status": "Aviso: Métricas inválidas (poucas previsões válidas)." })
                continue

            # 4. Prever Último Dia e Sinal (Chama ml_pipeline.py)
            prob = prever_ultimo_dia(df, modelos, scaler, features)
            sinal_binario = "BUY" if prob > entry_thr else "HOLD/SELL"

            # 5. Backtests (Chama backtester.py)
            bt = backtest_simples(df, oof_pred, limiar=entry_thr, horizonte=horizonte)
            bt_adv = backtest_avancado(df, oof_pred, limiar=entry_thr, horizonte=horizonte, atr_mult=atr_mult)

            # 6. Adicionar Resultado
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
            ).model_dump())

        except Exception as e:
            # Captura e registra erros internos
            print(f"  ❌ Erro catastrófico ao processar {ticker}: {e}")
            resultados_lista.append({"ticker": ticker, "status": f"Erro interno: {type(e).__name__}"})


    return resultados_lista

# ===============================================
# ENDPOINTS DA API (Rotas FastAPI)
# ===============================================

@app.get("/")
async def root():
    """Endpoint de saúde para verificar se a API está online."""
    return {"status": "online", "service": "Quant Trading Analysis API (Modular)"}

@app.post("/analyze_signals", response_model=List[SignalResult])
async def analyze_signals(request: SignalRequest):
    """
    Executa a análise completa de Machine Learning (Dados -> Features -> Treino Walk-Forward -> Sinal) 
    para a lista de ativos fornecida.
    """
    print(f"Requisição recebida para {len(request.tickers)} ativos.")
    
    # Chama a função de orquestração com os parâmetros da requisição
    results = processar_ativos_api(
        tickers=request.tickers,
        start_date=request.start_date,
        end_date=request.end_date,
        horizonte=request.horizonte_dias,
        k=request.k_volatilidade,
        entry_thr=request.entry_threshold,
        atr_mult=request.atr_multiplier
    )

    # O FastAPI serializa automaticamente a lista de objetos SignalResult para JSON
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
```eof

Agora o seu `app/main.py` está puramente como o **Orquestrador**. Ele depende da existência dos módulos separados (`data_loader.py`, `feature_eng.py`, etc.).

Qual módulo você gostaria de confirmar ou editar agora? Sugiro o `app/data_loader.py`.
