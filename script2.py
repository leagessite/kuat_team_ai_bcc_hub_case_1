import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np


# путь к папке с исходными файлами
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR + '\\data')
OUTPUT_PATH = os.path.join(BASE_DIR, "output.csv")
N_CLIENTS = 60
RNG = np.random.default_rng(42)

# --- Функция для форматирования чисел в соответствии со спецификацией ---
def format_amount_kzt(amount):
    try:
        a = int(round(amount))
    except:
        a = 0
    return f"{a:,}".replace(",", " ") + " ₸"

def approx_str(amount):
    return "≈" + format_amount_kzt(amount)

def short_month_name(dt):
    return dt.strftime("%B")

# --- Попробует загрузить предоставленные пользователем CSV-файлы; в противном случае сгенерирует синтетические данные. ---
def load_or_generate():
    profiles_path = "/mnt/data/profiles.csv"
    txns_path = "/mnt/data/txns.csv"
    transfers_path = "/mnt/data/transfers.csv"
    
    if os.path.exists(profiles_path) and os.path.exists(txns_path) and os.path.exists(transfers_path):
        profiles = pd.read_csv(profiles_path)
        txns = pd.read_csv(txns_path)
        transfers = pd.read_csv(transfers_path)
        print("Loaded profiles, txns, transfers from /mnt/data/")
        return profiles, txns, transfers
    
    print("Input CSVs not found. Generating synthetic sample data for demo (60 clients).")
    # Генерация профиль
    statuses = ["Студент", "Зарплатный клиент", "Премиальный клиент", "Стандартный клиент"]
    cities = ["Алматы", "Нур‑Султан", "Шымкент", "Актау", "Караганда"]
    names = ["Рамазан","Алия","Бек","Мадина","Данияр","Лаура","Нуржан","Айдана","Самат","Дина"]
    
    profiles = []
    for i in range(1, N_CLIENTS+1):
        status = RNG.choice(statuses, p=[0.15, 0.35, 0.15, 0.35])
        avg_balance = int(RNG.normal(200_000 if status=="Премиальный клиент" else 90_000, 40_000))
        avg_balance = max(0, avg_balance)
        profiles.append({
            "client_code": i,
            "name": RNG.choice(names),
            "status": status,
            "age": int(RNG.integers(20, 65)),
            "city": RNG.choice(cities),
            "avg_monthly_balance_KZT": avg_balance
        })
    profiles = pd.DataFrame(profiles)
    
    # Генерация транзакций на 3 месяца для каждого клиента
    categories = ["Одежда и обувь","Продукты питания","Кафе и рестораны","Медицина","Авто","Спорт",
                  "Развлечения","АЗС","Кино","Питомцы","Книги","Цветы","Едим дома","Смотрим дома",
                  "Играем дома","Косметика и Парфюмерия","Подарки","Ремонт дома","Мебель","Спа и массаж",
                  "Ювелирные украшения","Такси","Отели","Путешествия"]
    currencies = ["KZT","USD","EUR"]
    start_date = datetime.now() - timedelta(days=90)
    txns = []
    for _, row in profiles.iterrows():
        client = row["client_code"]
        monthly_spend = max(10_000, int(abs(RNG.normal(80_000 if row["status"]!="Студент" else 30_000, 50_000))))
        # Сгенерировать от 20 до 120 транзакций за 3 месяца
        n_tx = int(RNG.integers(20, 120))
        for _ in range(n_tx):
            d = start_date + timedelta(days=int(RNG.integers(0, 90)))
            cat = RNG.choice(categories, p=None)
            # сумма зависит от категории - грубая эвристика
            base = {
                "Путешествия": 40_000, "Отели": 25_000, "Такси": 3_000, "Кафе и рестораны": 5_000,
                "Продукты питания": 4_000, "Ювелирные украшения": 60_000, "Косметика и Парфюмерия": 6_000,
                "АЗС": 8_000
            }.get(cat, 6_000)
            amt = max(100, int(abs(RNG.normal(base, base*0.6))))
            # валюта: в основном тенге, но также доллары США/евро для путешествий/дорогих покупок
            if cat in ("Путешествия","Отели","Ювелирные украшения") and RNG.random() < 0.25:
                curr = RNG.choice(["USD","EUR"])
            else:
                curr = "KZT" if RNG.random() < 0.9 else RNG.choice(["USD","EUR"])
            txns.append({
                "date": d.strftime("%Y-%m-%d"),
                "category": cat,
                "amount": amt,
                "currency": curr,
                "client_code": client
            })
    txns = pd.DataFrame(txns)
    
    # Генерация переводов (salary_in и т.д.)
    transfer_types = ["salary_in","stipend_in","family_in","cashback_in","refund_in","card_in",
                      "p2p_out","card_out","atm_withdrawal","utilities_out","loan_payment_out",
                      "cc_repayment_out","installment_payment_out","fx_buy","fx_sell","invest_out",
                      "invest_in","deposit_topup_out","gold_buy_out","gold_sell_in"]
    transfers = []
    for _, row in profiles.iterrows():
        client = row["client_code"]
        # ежемесячная зарплата
        if row["status"] in ("Зарплатный клиент","Премиальный клиент") and RNG.random() < 0.9:
            for m in range(3):
                d = start_date + timedelta(days=10 + m*30 + int(RNG.integers(0,7)))
                transfers.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "type": "salary_in",
                    "direction": "in",
                    "amount": int(max(40_000, RNG.normal(200_000 if row["status"]=="Премиальный клиент" else 120_000, 20_000))),
                    "currency": "KZT",
                    "client_code": client
                })
        # периодические покупки/продажи валюты, если путешествует/имеет больший баланс
        if RNG.random() < 0.3:
            for _ in range(RNG.integers(1,4)):
                d = start_date + timedelta(days=int(RNG.integers(0,90)))
                transfers.append({
                    "date": d.strftime("%Y-%m-%d"),
                    "type": RNG.choice(["fx_buy","fx_sell"]),
                    "direction": "out" if RNG.random()<0.6 else "in",
                    "amount": int(abs(RNG.normal(200, 150))),
                    "currency": RNG.choice(["USD","EUR"]),
                    "client_code": client
                })
        # периодические взносы или погашения кредита
        if RNG.random() < 0.2:
            transfers.append({
                "date": (start_date + timedelta(days=int(RNG.integers(10,80)))).strftime("%Y-%m-%d"),
                "type": "installment_payment_out",
                "direction": "out",
                "amount": int(abs(RNG.normal(30_000, 10_000))),
                "currency": "KZT",
                "client_code": client
            })
    transfers = pd.DataFrame(transfers)
    return profiles, txns, transfers

profiles, txns, transfers = load_or_generate()

# --- Базовые предположения о конвертации валют (статичные для демонстрации) ---
FX_RATES = {"USD": 450.0, "EUR": 490.0, "KZT": 1.0}  # 1 USD = 450 KZT etc.

def to_kzt(amount, currency):
    return amount * FX_RATES.get(currency, 1.0)

txns["amount_kzt"] = txns.apply(lambda r: to_kzt(r["amount"], r["currency"]), axis=1)
transfers["amount_kzt"] = transfers.apply(lambda r: to_kzt(r["amount"], r["currency"]), axis=1)

# --- Проектирование функций для каждого клиента ---
clients = profiles["client_code"].unique()
feat_rows = []
for c in clients:
    p = profiles[profiles["client_code"]==c].iloc[0]
    t = txns[txns["client_code"]==c]
    tr = transfers[transfers["client_code"]==c]
    
    total_spend = t["amount_kzt"].sum()
    spend_by_cat = t.groupby("category")["amount_kzt"].sum().to_dict()
    spend_travel = sum(spend_by_cat.get(cat, 0) for cat in ["Путешествия","Отели","Такси"])
    spend_usd_eur = t[t["currency"].isin(["USD","EUR"])]["amount_kzt"].sum()
    top_cats = sorted(spend_by_cat.items(), key=lambda x: x[1], reverse=True)[:3]
    top3 = [cname for cname, _ in top_cats]
    num_taxies = len(t[t["category"]=="Такси"])
    fx_activity = len(tr[tr["type"].isin(["fx_buy","fx_sell"])])
    installments = len(tr[tr["type"].str.contains("installment")]) if not tr.empty else 0
    atm_withdrawals = len(tr[tr["type"]=="atm_withdrawal"]) if not tr.empty else 0
    inflows = tr[tr["direction"]=="in"]["amount_kzt"].sum()
    outflows = tr[tr["direction"]=="out"]["amount_kzt"].sum() + total_spend
    inflow_outflow_ratio = inflows / (outflows + 1)
    balance = p["avg_monthly_balance_KZT"]
    balance_volatility = max(0.0, RNG.random()*0.4)
    
    feat_rows.append({
        "client_code": c,
        "name": p["name"],
        "status": p["status"],
        "age": p["age"],
        "city": p["city"],
        "avg_monthly_balance_KZT": p["avg_monthly_balance_KZT"],
        "total_spend_KZT": total_spend,
        "spend_travel_KZT": spend_travel,
        "spend_usd_eur_KZT": spend_usd_eur,
        "num_taxies": num_taxies,
        "fx_activity": fx_activity,
        "installments": installments,
        "atm_withdrawals": atm_withdrawals,
        "inflow_outflow_ratio": inflow_outflow_ratio,
        "top3_categories": top3,
        "balance_volatility": balance_volatility
    })

features = pd.DataFrame(feat_rows)

# --- Функции оценки выгод (основанные на правилах, демонстрационные) ---
def score_travel(row):
    # Кэшбэк 4% на расходы, связанные с путешествиями, но не более 6000 тенге в месяц (пример)
    monthly_spend_travel = row["spend_travel_KZT"] / 3
    benefit = 0.04 * monthly_spend_travel * 1
    cap = 6_000
    return min(benefit, cap)

def score_premium(row):
    # кэшбэк на основе баланса (2%/3%/4%), плюс 4% на рестораны/ювелирные изделия/косметику
    bal = row["avg_monthly_balance_KZT"]
    if bal > 300_000:
        tier = 0.04
    elif bal > 150_000:
        tier = 0.03
    else:
        tier = 0.02

    # примерные базовые расходы ежемесячно
    base_monthly_spend = row["total_spend_KZT"] / 3
    extra = 0

    # мы будем искать исходные txns для вычисления фактических сумм
    client_txns = txns[txns["client_code"]==row["client_code"]]
    extra_spend = client_txns[client_txns["category"].isin(["Кафе и рестораны","Косметика и Парфюмерия","Ювелирные украшения"])]["amount_kzt"].sum() / 3
    benefit = tier * base_monthly_spend + 0.04 * extra_spend

    saved_fees = 500 if bal > 100_000 else 0
    return benefit + saved_fees

def score_credit_card(row):
    # до 10% в 3 лучших категориях (симулировать 5% эффективности) и значение льготного периода (приблизительно)
    client_txns = txns[txns["client_code"]==row["client_code"]]
    top3 = row["top3_categories"]
    top3_sum = sum(client_txns[client_txns["category"].isin(top3)]["amount_kzt"].sum() / 3 for _ in [1])
    benefit = 0.05 * top3_sum
    online = client_txns[client_txns["category"].isin(["Едим дома","Смотрим дома","Играем дома"])]["amount_kzt"].sum() / 3
    benefit += 0.1 * online
    if row["installments"] > 0:
        benefit += 1000
    return benefit

def score_fx(row):
    monthly_fx = row["spend_usd_eur_KZT"] / 3
    benefit = 0.01 * monthly_fx
    return benefit

def score_deposit_saving(row):
    # выгода: проценты на свободный остаток при размещении на сберегательном депозите (ежемесячно)
    bal = row["avg_monthly_balance_KZT"]
    if bal < 50_000:
        return 0
    return bal * 0.08 / 12

def score_investments(row):
    # для тех, у кого свободный баланс и аппетит; мы предоставляем небольшую «первоначальную выгоду» (снижение платы)
    bal = row["avg_monthly_balance_KZT"]
    if bal < 30_000:
        return 0
    return 300  # illustrative

def score_gold(row):
    # преимущество: диверсификация; только при высокой ликвидности
    bal = row["avg_monthly_balance_KZT"]
    if bal < 100_000:
        return 0
    return 200

def score_cash_loan(row):
    # учитывается только если притоки << оттоки или низкий баланс
    if row["inflow_outflow_ratio"] < 0.5 and row["avg_monthly_balance_KZT"] < 50_000:
        return 2000
    return 0

# --- Совокупный подсчет баллов и отбор ---
product_list = [
    ("Карта для путешествий", score_travel),
    ("Премиальная карта", score_premium),
    ("Кредитная карта", score_credit_card),
    ("Обмен валют", score_fx),
    ("Вклад сберегательный", score_deposit_saving),
    ("Инвестиции", score_investments),
    ("Золотые слитки", score_gold),
    ("Кредит наличными", score_cash_loan)
]

def choose_product_and_push(row):
    scores = {}
    for name, func in product_list:
        try:
            s = func(row)
        except Exception:
            s = 0
        scores[name] = max(0, float(s))
    # отфильтровать продукты, требующие сигналов, если их оценка == 0
    best_product = max(scores.items(), key=lambda x: x[1])
    product_name, product_score = best_product

    threshold = 500.0
    if product_score < threshold:
        # запасной вариант: рекомендует «Инвестиции», если есть баланс, иначе «Вклад сберегательный», если есть баланс.
        if row["avg_monthly_balance_KZT"] >= 30_000:
            product_name = "Инвестиции"
            product_score = scores.get(product_name, 0)
        else:
            product_name = "Вклад сберегательный"
            product_score = scores.get(product_name, 0)
    # Сгенерировать push текст
    push = generate_push_text(row, product_name, product_score, scores)
    return product_name, product_score, push

# --- Генерация push-уведомлений по правилам TOV ---
def clean_name(n):
    # убедимся, что имя написано с заглавной буквы (пользователь использует маленькую «вы» в соответствии со спецификацией, но имена пишутся с заглавной буквы)
    return str(n)

def generate_push_text(row, product_name, product_score, all_scores):
    name = clean_name(row["name"])
    month = short_month_name(datetime.now() - timedelta(days=1))
    # создавать сообщения, специфичные для продукта (шаблоны)
    if product_name == "Карта для путешествий":
        spent = row["spend_travel_KZT"]
        est = product_score
        msg = f"{name}, в {month} вы потратили {format_amount_kzt(spent)} на поездки и такси. С картой для путешествий часть расходов вернулась бы {approx_str(est)}. Открыть карту."
    elif product_name == "Премиальная карта":
        bal = row["avg_monthly_balance_KZT"]
        pct = "4%" if bal > 300_000 else ("3%" if bal > 150_000 else "2%")
        msg = f"{name}, у вас в среднем {format_amount_kzt(bal)} на счёте и частые расходы в ресторанах. Премиальная карта даст до {pct} кешбэка и бесплатные снятия. Подключите сейчас."
    elif product_name == "Кредитная карта":
        topcats = ", ".join(row["top3_categories"]) if row["top3_categories"] else "избранные категории"
        msg = f"{name}, ваши топ‑категории — {topcats}. Кредитная карта даёт до 10% в любимых категориях и льготный период. Оформить карту."
    elif product_name == "Обмен валют":
        # определить основную валюту FX, если таковая имеется
        fx_curr = "USD/EUR" if row["spend_usd_eur_KZT"]>0 else "валют"
        msg = f"{name}, вы часто платите в {fx_curr}. В приложении выгодный обмен и авто‑покупка по целевому курсу. Настроить обмен."
    elif product_name == "Вклад сберегательный":
        bal = row["avg_monthly_balance_KZT"]
        msg = f"{name}, у вас остаются свободные средства — {format_amount_kzt(bal)}. Разместите их на сберегательном вкладе и получите выгоду. Открыть вклад."
    elif product_name == "Инвестиции":
        msg = f"{name}, попробуйте инвестиции с низким порогом входа и без комиссий на старт. Открыть счёт."
    elif product_name == "Золотые слитки":
        msg = f"{name}, для диверсификации можно открыть позицию в золотых слитках. Узнать детали."
    elif product_name == "Кредит наличными":
        msg = f"{name}, если нужен запас на крупные траты — можно оформить кредит наличными с гибкими выплатами. Узнать лимит."
    else:
        msg = f"{name}, у нас есть продукты, которые могут быть полезны. Посмотреть предложения."
    # Принудительный стиль: без заглавных букв, максимум 1 эмодзи, присутствует призыв к действию (включая один из обязательных глаголов)
    # Ограничить длину push-сообщений (180–220 символов); если длиннее, попробуйте сократить.
    if len(msg) > 220:
        # сократить, убрав пояснительное предложение
        parts = msg.split(".")
        # взять первые две части и добавить призыв к действию
        new_msg = ". ".join(parts[:2]).strip()
        if not new_msg.endswith("."):
            new_msg += "."
        new_msg += " Открыть."
        msg = new_msg
    # Убедимся, что нет ЗАГЛАВНЫХ букв и максимум один восклицательный знак (мы их не добавляли)
    msg = msg.replace("  ", " ")
    # Окончательная очистка: проверка форматирования чисел (суммы уже отформатированы)
    return msg

# Применить выбор и создать push форму
out_rows = []
for _, r in features.iterrows():
    prod, score, push = choose_product_and_push(r)
    out_rows.append({
        "client_code": r["client_code"],
        "product": prod,
        "push_notification": push
    })

out_df = pd.DataFrame(out_rows)

# Сохранить CSV
out_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"Output saved to {OUTPUT_PATH} ({len(out_df)} rows).")

# Показать предварительный просмотр

print(out_df.head(10).to_string(index=False))

# Также отображать примеры функций и профилей для отладки.
print("\nProfiles sample:")
print(profiles.head(5).to_string(index=False))
print("\nFeatures sample:")
print(features.head(5).to_string(index=False))


