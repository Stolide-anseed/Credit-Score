# Credit Score Prediction

ML-проект для предсказания кредитного рейтинга клиента: `Poor`, `Standard` или `Good`.
Внутри есть полный путь от EDA и feature engineering до финальной LightGBM-модели,
сохраненного preprocessing pipeline и FastAPI-сервиса для inference.

## Коротко о результате

Это не просто учебный классификатор, а аккуратно докрученный скоринговый пайплайн:

- смог поднять `Average Precision` для самого сложного класса `Good` с `56.89%` до `81.59%`, то есть примерно в `1.43` раза;
- поднял `precision` класса `Good` с `49.22%` до `76.26%`, почти в `1.55` раза;
- вывел `ROC-AUC OVR` для `Good` с `87.96%` до `96.76%`;
- получил `87.08%` по целевой Optuna-метрике `mean_average_precision_priority_classes`;
- добился `92.58% Average Precision` для `Standard` и `81.59%` для `Good`, то есть модель хорошо держит не только самый частый класс.

Главная ставка была на инженерные финансовые признаки: долговая нагрузка, давление ставки
на долг, кредитные запросы относительно истории, структура кредитных продуктов и признаки
платежного поведения. В итоговой LightGBM-модели `engineered_numeric` признаки дают `55.41%`
gain importance, а топовый признак `interest_debt_pressure` один закрывает `14.66%` gain.

## Метрики

Сравнение моделей по ключевым классам и классу `Good`, который важен из-за меньшей доли в данных:

| Модель | AP Standard | AP Good | Precision Good | Recall Good | ROC-AUC Good |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.8035 | 0.5689 | 0.4922 | 0.8307 | 0.8796 |
| LightGBM | 0.8865 | 0.7410 | 0.5613 | 0.8781 | 0.9415 |
| XGBoost | 0.8647 | 0.6758 | 0.5194 | 0.8594 | 0.9185 |
| CatBoost | 0.8697 | 0.6982 | 0.5286 | 0.8621 | 0.9269 |
| LightGBM + Optuna | 0.9258 | 0.8159 | 0.7626 | 0.8800 | 0.9676 |

Графики и таблицы лежат в `Metrics/`:

- `Metrics/BaseLine_Metrics.png`
- `Metrics/LightGbm.png`
- `Metrics/XGBoost.png`
- `Metrics/Catboost.png`
- `Metrics/LightGBM_with_tuning_hyperparametrs.png`
- `Metrics/Pr-curve/`

## Данные

Исходный датасет: `data/train.csv`.

- `100 000` строк;
- `28` исходных колонок;
- `12 500` клиентов;
- период наблюдений с `January` по `August`;
- таргет `Credit_Score`: `Poor`, `Standard`, `Good`.

Для честной проверки используется split по `Customer_ID`: клиент не может попасть одновременно
в train и test. Текущий split:

- train: `80 000` строк, `10 000` клиентов;
- test: `20 000` строк, `2 500` клиентов;
- customer overlap: `0`;
- customer profile test overlap: `0`.

## Что внутри

```text
app/
  main.py                         FastAPI inference service
scr/
  preprocessing.py                очистка, feature engineering, sklearn pipeline
  lightgbm_optuna_search.py       Optuna-подбор LightGBM
  train_lightgbm_final.py         обучение финальной модели
  lightgbm_feature_selection.py   отчет по важности признаков
notebooks/
  EDA.ipynb
  preprocessing.ipynb
  catboost_experiment.ipynb
  xgboost_experiment.ipynb
  lightgbm_experiment.ipynb
model/
  model.joblib                    финальная LightGBM-модель
  preprocessor.joblib             сохраненный preprocessing pipeline
  model_metadata.json             параметры, признаки и метаданные обучения
Metrics/
  *.png                           таблицы метрик и PR-кривые
  feature_selection/              отчет по feature importance
```

## Подход

1. EDA показал качество данных, пропуски, выбросы и дисбаланс классов.
2. Preprocessing чистит строковые числа, служебные маркеры пропусков, аномалии и PII-поля.
3. Split строится по `Customer_ID`, чтобы не было leakage между train и test.
4. Feature engineering добавляет финансовые ratio-признаки, loan type признаки, ordinal-кодировки и flags.
5. Сравниваются baseline, CatBoost, XGBoost и LightGBM.
6. Финальная LightGBM донастроена через Optuna на `Average Precision` для приоритетных классов `Standard` и `Good`.
7. Модель и препроцессор сохранены как `joblib` и завернуты в FastAPI.

## Установка

Файл зависимостей в проекте называется `requrements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requrements.txt
```

## Запуск API

```powershell
uvicorn app.main:app --reload
```

После запуска:

- Swagger UI: `http://127.0.0.1:8000/docs`
- health check: `http://127.0.0.1:8000/health`
- prediction endpoint: `POST http://127.0.0.1:8000/predict`

Пример запроса:

```json
{
  "features": {
    "Customer_ID": "CUS_0x1234",
    "Month": "January",
    "Age": "35",
    "Occupation": "Engineer",
    "Annual_Income": "76000",
    "Monthly_Inhand_Salary": "6200",
    "Num_Bank_Accounts": 4,
    "Num_Credit_Card": 5,
    "Interest_Rate": 12,
    "Num_of_Loan": "2",
    "Type_of_Loan": "Auto Loan, Credit-Builder Loan",
    "Delay_from_due_date": 5,
    "Num_of_Delayed_Payment": "2",
    "Changed_Credit_Limit": "7.5",
    "Num_Credit_Inquiries": 3,
    "Credit_Mix": "Standard",
    "Outstanding_Debt": "1250.40",
    "Credit_Utilization_Ratio": 31.2,
    "Credit_History_Age": "12 Years and 4 Months",
    "Payment_of_Min_Amount": "No",
    "Total_EMI_per_month": 180.5,
    "Amount_invested_monthly": "350",
    "Payment_Behaviour": "High_spent_Medium_value_payments",
    "Monthly_Balance": "420.75"
  },
  "return_probabilities": true
}
```

## Docker

```powershell
docker build -t credit-score-api .
docker run --rm -p 8000:8000 credit-score-api
```

## Воспроизведение обучения

Если prepared CSV уже есть в `data/`, можно сразу запускать подбор и финальное обучение:

```powershell
python scr/lightgbm_optuna_search.py --n-trials 50 --n-splits 5
python scr/train_lightgbm_final.py
python scr/lightgbm_feature_selection.py
```

Если нужно пересобрать preprocessing из ноутбука:

```powershell
cd scr
python run_preprocessing_notebook.py
cd ..
```

## Артефакты финальной модели

`model/model_metadata.json` фиксирует:

- модель: `lightgbm.LGBMClassifier`;
- число признаков: `107`;
- Optuna best trial: `34`;
- Optuna best value: `0.8708405962254729`;
- priority classes: `1` и `2`;
- class counts train: `Poor` - `23272`, `Standard` - `42594`, `Good` - `14134`.

## Вывод

Проект доведен до состояния, где модель можно не только показать в ноутбуке, но и поднять как сервис.
Сильная часть решения - не один алгоритм, а связка: честный split без клиентского leakage,
финансово интерпретируемые признаки, Optuna-тюнинг под нужные бизнес-классы и готовый inference API.
