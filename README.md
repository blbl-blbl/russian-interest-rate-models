# Russian Interest Rate Models

Исследовательский проект: исторические процентные ставки России, модели Vasicek и CIR и проверка их прогнозной способности.

## Установка

Требуется Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Загрузка данных

```bash
# Все три ряда, с 01.01.2014 по текущую дату:
python -m russian_rates

# Только RUONIA за конкретный период:
python -m russian_rates --series ruonia --start 2014-01-01 --end 2026-11-20

# Кривая бескупонной доходности ОФЗ:
python -m russian_rates --series zcyc --start 2022-01-01

# Заменить накопленную историю только запрошенным периодом:
python -m russian_rates --series key_rate --start 2020-01-01 --replace
```

Принимаются даты `YYYY-MM-DD` и `DD.MM.YYYY`. По умолчанию данные сохраняются в `data/processed/` (не включается в Git). Запросы разбиваются по календарным годам, ограничены по времени и повторяются при временных HTTP-ошибках (429/5xx). Парсер отвергает неожиданные структуры и неверные числа вместо молчаливого сохранения повреждённых данных.

Повторная загрузка **дополняет существующую историю**: совпадающие даты/сроки заменяются новыми значениями. `--replace` полностью заменяет историю выбранным периодом. Запись осуществляется через временный файл с последующей атомарной заменой.

| Parquet | Поля |
| --- | --- |
| `ruonia.parquet` | `date`, `ruonia_pct`, `volume_bln_rub` |
| `key_rate.parquet` | `date`, `key_rate_pct` |
| `zcyc.parquet` | `date`, `tenor_years`, `yield_pct` |

КБД хранится в **длинном формате**: по 12 сроков от 0,25 до 30 лет на дату. Даты имеют тип `datetime64[ns]`, числовые показатели — `float64`. Все ставки и доходности — **в процентах годовых**, например `14.25`, а не `0.1425`; для стохастических моделей их обычно делят на 100. В данных сохраняются даты наблюдений; пропуски выходных и праздников автоматически не заполняются. Для RUONIA отсутствующий объём сделок (символ `—`, `–` или `-` в ответе ЦБ) сохраняется как `NaN`; сама ставка остаётся обязательной. Нули и значения с соседних дат не подставляются.

## Использование в Python

```python
import pandas as pd

ruonia = pd.read_parquet("data/processed/ruonia.parquet")
ruonia["ruonia_decimal"] = ruonia["ruonia_pct"] / 100

zcyc = pd.read_parquet("data/processed/zcyc.parquet")
curve = zcyc.pivot(index="date", columns="tenor_years", values="yield_pct")
```

## Тестирование

```bash
python -m pytest -q
```

Тесты используют фиксированные HTML-примеры и имитацию HTTP-ответов; реальный сетевой доступ к ЦБ не нужен. После установки пакета рекомендуется отдельно выполнить короткую загрузку за несколько дней, чтобы проверить текущее состояние сайта и поддержку Parquet локальной версией `pyarrow`.

Источники Банка России: [RUONIA](https://www.cbr.ru/hd_base/ruonia/dynamics/), [ключевая ставка](https://www.cbr.ru/hd_base/keyrate/), [КБД ОФЗ](https://www.cbr.ru/hd_base/zcyc_params/).
