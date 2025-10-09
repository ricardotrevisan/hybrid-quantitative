### Scope - Hybrid logic

* O GARCH fornece a volatilidade diária (risco/ruído).
* Essa volatilidade entra como feature extra na LSTM (assim como outras em modelos produtivos)
* A LSTM aprende padrões não lineares combinando preço e risco - eficaz para sequências
* Resultado: previsão mais robusta do que apenas LSTM ou GARCH isolados.

---

### Adj Close:

#### Dividendo e desdobramentos:
* O Adj Close ajusta o preço para refletir dividendos pagos, splits (desdobramentos) e outras alterações corporativas.
* Isso evita que mudanças artificiais no preço confundam o modelo.

#### Comparação histórica consistente:
* Quando usamos Close puro, picos ou quedas podem aparecer artificialmente devido a dividendos ou splits.
* Adj Close garante que os retornos calculados representem verdadeiramente a valorização do ativo.

#### Retornos logarítmicos mais precisos:
* Se você calcula log-retornos ou pct_change para alimentar o GARCH ou LSTM, usar Adj Close dá valores consistentes e não distorcidos.


### TensorFlow Check:
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

### TODO
* Janela deslizante com previsão iterativa - 30, 45, 50 e 60 dias ✔️ 
    * 60 dias com melhores resultados com as Features atuais
* Otimimzar epochs x janela ✔️
* Adicionar 3a feature, RSI (sensibilidade a reversões com momentum de preço) ✔️
    * Separar scalers por feature (Close, Volatility, RSI) para preservar a escala natural de cada indicador.✔️
* Pipeline Basico:
    * Close, Open, High, Low e Return
    * Lags (20 dias) - capturar autocorrelação✔️
    * Médias Móveis (EMA) (a boa e velha... 9, 21 e 50 dias)✔️
    * Indicadores 
        * RSI (14)✔️
        * MACD (12,26,9)✔️
            * Signal Line ✔️
            * Histogram ✔️
        * Momentum 3/6/12 dias
        * Bollinger Bands
        * ATR (14)
        * OBV / VWAP
        * Bid-as spread, depth
        * VIX
        * Drawdown local (distância do preço até a última máxima)

* RTD/Profit Pro (Forçando Bridge via VBA/Excel)✔️
* Engenharia de Dados : multicolinearidade (ficou sério agora! 😂)
    * Rolling z-score (window = 60), pra reduzir efeitos de descontinuidade causados pelos cisnes (COVID, já que a base vem de jan/2020) - vai dizer o quanto o retorno está fora do padrão "recente", irmão!✔️
    * As lags nao ajudam um LSTM (L, de long memory.... parece só terem poluído a coisa.... !redundância temporal, aumento de colinearidade que degrada generalização, melhor dizendo.... )
    👁️* Weights maiores a dados recentes : decay exponencial, pode ajudar
    * LSTM 64-128, 1-2 camadas
    * batch_size 32-128 (chaos!)
    * dropout + recurrent_dropout: 0.2-0.4
    * weight decay (L2) 1e-5–1e-3

* Variação de modelos LSTM, GRU e Random Forest
* Avaliação:
* Defasagem temporal nas previsões, seguindo o preço, não está pegando picos e quedas abrutas
    * Features adicionais: volume e novos indicadores
    * Walk-forward validation: para aferir estabilidade temporal do modelo
    * Teste ensemble
    * Testar CNN-LSTM