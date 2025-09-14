import os
import pandas as pd
import re

# путь к папке с исходными файлами
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FOLDER = os.path.join(BASE_DIR)
DATA_FOLDER = FOLDER + '\\data'
txns_list = []
transfers_list = []

for file in os.listdir(DATA_FOLDER):
    if file.endswith("_transactions_3m.csv"):
        client_code = int(re.search(r"client_(\d+)_", file).group(1))
        df = pd.read_csv(os.path.join(DATA_FOLDER, file))
        df["client_code"] = client_code
        txns_list.append(df)

    elif file.endswith("_transfers_3m.csv"):
        client_code = int(re.search(r"client_(\d+)_", file).group(1))
        df = pd.read_csv(os.path.join(DATA_FOLDER, file))
        df["client_code"] = client_code
        transfers_list.append(df)

# объединяем в один DataFrame
txns = pd.concat(txns_list, ignore_index=True)
transfers = pd.concat(transfers_list, ignore_index=True)

# сохраняем
txns.to_csv(os.path.join(FOLDER, "txns.csv"), index=False, encoding="utf-8-sig")
transfers.to_csv(os.path.join(FOLDER, "transfers.csv"), index=False, encoding="utf-8-sig")

# копируем clients.csv как profiles.csv (если у тебя такой файл есть в папке)
clients_path = os.path.join(DATA_FOLDER, "clients.csv")
if os.path.exists(clients_path):
    profiles = pd.read_csv(clients_path)
    profiles.to_csv(os.path.join(FOLDER, "profiles.csv"), index=False, encoding="utf-8-sig")

print("Готово! Файлы profiles.csv, txns.csv, transfers.csv сохранены в папке case 1.")
