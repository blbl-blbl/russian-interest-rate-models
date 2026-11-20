# data_parser
# Парсинг данных по RUONIA, Ключевой ставке и Кривой доходности ОФЗ

import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
out_path = ROOT_DIR / "data"
out_path.mkdir(parents=True, exist_ok=True)

def get_ruonia(
        start_date:str = "01.01.2014",
        end_date:str = str(datetime.date.today().strftime("%d.%m.%Y"))
):
    url = f"https://www.cbr.ru/hd_base/ruonia/dynamics?UniDbQuery.Posted=True&UniDbQuery.From={start_date}&UniDbQuery.To={end_date}"
    response = requests.get(url)

    if response.status_code != 200:
        print("Не удалось установить соединение с сайтом ЦБ (RUONIA)")
        return 0

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("div", class_="table-wrapper table")
        rows = table.find_all("tr")

        # Названия столбцов
        # first_row = rows[0]
        # table_titles = [column.text.strip() for column in first_row.find_all("th")[:3]]


        dates = []
        ruonias = []
        values = []

        for row_number in range(1, len(rows)):
            columns = [column.text.strip() for column in rows[row_number].find_all("td")[:3]]
            dates.append(columns[0])
            ruonias.append(columns[1])
            values.append(columns[2])

    except ValueError as er:
        raise f"Возникла ошибка при парсинге RUONIA, {er}"


    try:
        df = pd.DataFrame(
            {
                "Дата": dates,
                "Ставка RUONIA, % годовых": ruonias,
                "Объем сделок RUONIA, млрд руб.": values
            }
        )
        df.to_excel(out_path / f"RUONIA_DATA_{start_date}_{end_date}.xlsx", index=False)

        return 1

    except ValueError as er:
        raise f"Возникла обшика при сохранении в Датафрейм RUONIA, {er}"


def get_zcyc_params(
        start_date:str = "01.01.2014",
        end_date:str = str(datetime.date.today().strftime("%d.%m.%Y"))
):
    url = f"https://www.cbr.ru/hd_base/zcyc_params/?UniDbQuery.Posted=True&UniDbQuery.From={start_date}&UniDbQuery.To={end_date}"
    response = requests.get(url)

    if response.status_code != 200:
        print("Не удалось установить соединение с сайтом ЦБ (КБД)")
        return 0

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("div", class_="table-wrapper")
        rows = table.find_all("tr")[1:] # пропускаем 1 строку - в ней ничего полезного

        # Названия столбцов
        years_to_maturity = [column.text.strip() for column in rows[0].find_all("th")]

        dates = rows[0]
        rate_025 = rows[1]
        rate_05 = rows[2]
        rate_075 = rows[3]
        rate_1 = rows[4]
        rate_2 = rows[5]
        rate_3 = rows[6]
        rate_5 = rows[7]
        rate_7 = rows[8]
        rate_10 = rows[9]
        rate_15 = rows[10]
        rate_20 = rows[11]
        rate_30 = rows[12]

        for row_number in range(1, len(rows)):
            columns = [column.text.strip() for column in rows[row_number].find_all("td")]
            dates.append(columns[0])
            rate_025.append(columns[1])
            rate_05.append(columns[2])
            rate_075.append(columns[3])
            rate_1.append(columns[4])
            rate_2.append(columns[5])
            rate_3.append(columns[6])
            rate_5.append(columns[7])
            rate_7.append(columns[8])
            rate_10.append(columns[9])
            rate_15.append(columns[10])
            rate_20.append(columns[11])
            rate_30.append(columns[12])

    except ValueError as er:
        raise f"Возникла ошибка при парсинге КБД, {er}"


    try:
        df = pd.DataFrame(
            {
                "Дата": dates,
                years_to_maturity[0]: rate_025,
                years_to_maturity[1]: rate_05,
                years_to_maturity[2]: rate_075,
                years_to_maturity[3]: rate_1,
                years_to_maturity[4]: rate_2,
                years_to_maturity[5]: rate_3,
                years_to_maturity[6]: rate_5,
                years_to_maturity[7]: rate_7,
                years_to_maturity[8]: rate_10,
                years_to_maturity[9]: rate_15,
                years_to_maturity[10]: rate_20,
                years_to_maturity[11]: rate_025,
                years_to_maturity[11]: rate_30,
            }
        )
        df.to_excel(out_path / f"ZCYC_{start_date}_{end_date}.xlsx", index=False)

        return 1

    except ValueError as er:
        raise f"Возникла обшика при сохранении в Датафрейм КБД, {er}"


def get_key_rate(
        start_date:str = "01.01.2014",
        end_date:str = str(datetime.date.today().strftime("%d.%m.%Y"))
):
    url = f"https://www.cbr.ru/hd_base/keyrate/?UniDbQuery.Posted=True&UniDbQuery.From={start_date}&UniDbQuery.To={end_date}"
    response = requests.get(url)

    if response.status_code != 200:
        print("Не удалось установить соединение с сайтом ЦБ (KEY RATE)")
        return 0

    try:
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("div", class_="table-wrapper")
        rows = table.find_all("tr")

        dates = []
        key_rates = []

        for row_number in range(1, len(rows)):
            columns = [column.text.strip() for column in rows[row_number].find_all("td")]
            dates.append(columns[0])
            key_rates.append(columns[1])

    except ValueError as er:
        raise f"Возникла ошибка при парсинге KEY RATE, {er}"


    try:
        df = pd.DataFrame(
            {
                "Дата": dates,
                "KEY RATE": key_rates,
            }
        )
        df.to_excel(out_path / f"KEY_RATE_{start_date}_{end_date}.xlsx", index=False)

    except ValueError as er:
        raise f"Возникла обшика при сохранении в Датафрейм KEY_RATE, {er}"

if __name__ == "__main__":
    get_ruonia()
    get_zcyc_params()
    get_key_rate()