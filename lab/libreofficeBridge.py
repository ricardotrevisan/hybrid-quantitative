import uno
import csv
from datetime import datetime
import time

# Conecta ao LibreOffice UNO (assume LibreOffice rodando com --accept=socket)
local_context = uno.getComponentContext()
resolver = local_context.ServiceManager.createInstanceWithContext(
    "com.sun.star.bridge.UnoUrlResolver", local_context)
ctx = resolver.resolve("uno:socket,host=localhost,port=2002;urp;StarOffice.ComponentContext")
smgr = ctx.ServiceManager
desktop = smgr.createInstanceWithContext("com.sun.star.frame.Desktop", ctx)
doc = desktop.getCurrentComponent()
sheet = doc.Sheets[0]

# Configuração
linhas_ativos = [1, 2]        # linhas dos ativos (1 = WDOX25, 2 = PETR4)
num_colunas = 10              # número de colunas: Data, Hora, Último, etc.
headers = ["Asset","Data","Hora","Último","Abertura","Máximo","Mínimo","Strike","Negócios","Vencimento"]
filename = "ativos_rtd.csv"
dados = []

try:
    while True:
        timestamp = datetime.now()
        for l in linhas_ativos:
            linha = {"timestamp": timestamp}
            for c in range(num_colunas):
                cell = sheet.getCellByPosition(c, l)  # (col, linha), linha 0 = primeira linha
                linha[headers[c]] = cell.Value if isinstance(cell.Value, (int,float)) else str(cell.String)
            dados.append(linha)
            print(linha)

        # Salva periodicamente
        if len(dados) >= 20:
            with open(filename, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["timestamp"] + headers)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(dados)
            dados = []
            print(f"20 linhas salvas em {filename}")

        time.sleep(1)

except KeyboardInterrupt:
    if dados:
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp"] + headers)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerows(dados)
    print(f"\nColeta interrompida. Dados finais salvos em {filename}")
