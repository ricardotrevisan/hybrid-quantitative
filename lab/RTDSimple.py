import win32com.client
import time

# Conecta ao RTD do Profit
rtd = win32com.client.Dispatch("RTDTrading.RTDServer")
time.sleep(3)  # espera inicial para o RTD carregar

# Ativo e campo para teste
ativo = "WDOX25_F_0"  # exemplo: dólar futuro
campo = "ULT"         # último preço

try:
    while True:
        try:
            valor = rtd.ConnectData(1, [ativo, campo, ""])
            print(f"{ativo} {campo}: {valor}")
        except Exception as e:
            print(f"Erro ao pegar {ativo} {campo}: {e}")

        time.sleep(1)  # coleta a cada 1 segundo

except KeyboardInterrupt:
    print("\nTeste interrompido pelo usuário.")


# Asset	Data	Hora	Último	Volume
# WDOX25	=RTD("RTDTrading.RTDServer";; "WDOX25_F_0"; "DAT")	=RTD("RTDTrading.RTDServer";; "WDOX25_F_0"; "HOR")	=RTD("RTDTrading.RTDServer";; "WDOX25_F_0"; "ULT")	=RTD("RTDTrading.RTDServer";; "WDOX25_F_0"; "VOL")
# PETR4	=RTD("RTDTrading.RTDServer";; "PETR4_B_0"; "DAT")	=RTD("RTDTrading.RTDServer";; "PETR4_B_0"; "HOR")	=RTD("RTDTrading.RTDServer";; "PETR4_B_0"; "ULT")	=RTD("RTDTrading.RTDServer";; "PETR4_B_0"; "VOL")
