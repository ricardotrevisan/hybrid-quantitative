import win32com.client
import pandas as pd
import time
from datetime import datetime

# Conecta ao RTD do Profit
rtd = win32com.client.Dispatch("RTDTrading.RTDServer")
time.sleep(1)  # espera inicial

# Ativos e campos que vamos capturar
ativos = ["WDOX25_F_0", "PETR4_B_0"]
campos = ["DAT", "HOR", "ULT", "ABE", "MAX", "MIN", "PEX", "NEG", "VEN"]

# DataFrame inicial
df = pd.DataFrame(columns=["timestamp", "Asset"] + campos)

# Salvar periodicamente a cada N linhas
save_interval = 20
filename = "profit_rtd.csv"

try:
    while True:
        timestamp = datetime.now()
        for ativo in ativos:
            linha = {"timestamp": timestamp, "Asset": ativo}
            for campo in campos:
                try:
                    valor = rtd.ConnectData(1, [ativo, campo, ""])
                except Exception as e:
                    valor = None
                    print(f"Erro ao pegar {ativo} {campo}: {e}")
                linha[campo] = valor
            df = pd.concat([df, pd.DataFrame([linha])], ignore_index=True)
            print(linha)

        # Salva periodicamente
        if len(df) >= save_interval:
            df.to_csv(filename, index=False, mode="a", header=not bool(df.shape[0]-save_interval))
            df = pd.DataFrame(columns=["timestamp", "Asset"] + campos)
            print(f"{save_interval} linhas salvas em {filename}")

        time.sleep(1)

except KeyboardInterrupt:
    # Salva dados restantes
    if not df.empty:
        df.to_csv(filename, index=False, mode="a", header=not bool(df.shape[0]))
    print(f"\nColeta interrompida. Dados finais salvos em {filename}")
