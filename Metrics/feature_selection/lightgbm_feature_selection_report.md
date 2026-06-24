# LightGBM feature selection

Создано UTC: `2026-06-24T14:30:12.885528+00:00`

## Контекст

- Модель: `C:\Users\stoli\PycharmProjects\Credit_score\model\model.joblib`
- Metadata: `C:\Users\stoli\PycharmProjects\Credit_score\model\model_metadata.json`
- Метрика Optuna: `mean_average_precision_priority_classes`
- Лучшее значение Optuna: `0.8708405962254729`
- Порог cumulative gain: `0.95`
- Всего признаков: `107`
- Выбрано признаков: `39`
- Покрытие gain выбранными признаками: `0.955782`

## Как читать отчёт

`gain_importance` показывает вклад признака в снижение ошибки в деревьях LightGBM. `split_importance` показывает, сколько раз признак использовался в разбиениях. Для feature selection используется минимальный набор признаков, который покрывает заданную долю cumulative gain.

Важно: это model-based selection, а не доказательство причинности. Коррелирующие признаки могут делить важность между собой.

## Top-30 признаков по gain

| rank_by_gain | feature | feature_family | gain_share | cumulative_gain_share | split_share | selected_by_cumulative_gain |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | interest_debt_pressure | engineered_numeric | 0.146642 | 0.146642 | 0.037784 | true |
| 2 | credit_mix_ordinal | engineered_numeric | 0.095054 | 0.241696 | 0.001692 | true |
| 3 | Outstanding_Debt | base_numeric | 0.053426 | 0.295122 | 0.029434 | true |
| 4 | Credit_Mix_Standard | credit_mix_ohe | 0.042696 | 0.337818 | 0.000637 | true |
| 5 | Changed_Credit_Limit | base_numeric | 0.034040 | 0.371858 | 0.043936 | true |
| 6 | Credit_Mix_Good | credit_mix_ohe | 0.029275 | 0.401133 | 0.000271 | true |
| 7 | Month_Num | month_ohe | 0.028992 | 0.430126 | 0.014042 | true |
| 8 | emi_to_monthly_salary | engineered_numeric | 0.028043 | 0.458169 | 0.032523 | true |
| 9 | Credit_History_Age_Months | base_numeric | 0.025973 | 0.484142 | 0.041220 | true |
| 10 | debt_per_credit_product | engineered_numeric | 0.024710 | 0.508852 | 0.036571 | true |
| 11 | Interest_Rate | base_numeric | 0.023607 | 0.532459 | 0.024322 | true |
| 12 | credit_inquiries_per_history_year | engineered_numeric | 0.023229 | 0.555688 | 0.033638 | true |
| 13 | Age | base_numeric | 0.022717 | 0.578405 | 0.035437 | true |
| 14 | Delay_from_due_date | base_numeric | 0.022523 | 0.600928 | 0.027315 | true |
| 15 | avg_delay_per_delayed_payment | engineered_numeric | 0.022168 | 0.623096 | 0.036909 | true |
| 16 | Total_EMI_per_month | engineered_numeric | 0.022136 | 0.645232 | 0.029977 | true |
| 17 | credit_age_per_loan | engineered_numeric | 0.020723 | 0.665955 | 0.033446 | true |
| 18 | Num_Credit_Card | base_numeric | 0.019378 | 0.685333 | 0.010460 | true |
| 19 | credit_cards_per_bank_account | engineered_numeric | 0.019014 | 0.704347 | 0.027383 | true |
| 20 | debt_to_annual_income | engineered_numeric | 0.018494 | 0.722841 | 0.025675 | true |
| 21 | Annual_Income | base_numeric | 0.017797 | 0.740638 | 0.023922 | true |
| 22 | delayed_payment_ratio | engineered_numeric | 0.017508 | 0.758146 | 0.032837 | true |
| 23 | inquiries_per_credit_account | engineered_numeric | 0.015985 | 0.774130 | 0.027724 | true |
| 24 | debt_to_monthly_salary | engineered_numeric | 0.015425 | 0.789555 | 0.022237 | true |
| 25 | Credit_Utilization_Ratio | base_numeric | 0.013819 | 0.803374 | 0.039482 | true |
| 26 | Monthly_Inhand_Salary | base_numeric | 0.013573 | 0.816947 | 0.017133 | true |
| 27 | available_income_after_emi | engineered_numeric | 0.012413 | 0.829360 | 0.015780 | true |
| 28 | Num_of_Delayed_Payment | base_numeric | 0.012383 | 0.841743 | 0.017139 | true |
| 29 | loans_per_bank_account | engineered_numeric | 0.012326 | 0.854069 | 0.019206 | true |
| 30 | Num_Bank_Accounts | base_numeric | 0.011622 | 0.865691 | 0.011874 | true |

## Важность по группам признаков

| feature_family | feature_count | selected_count | gain_share | split_share |
| --- | --- | --- | --- | --- |
| engineered_numeric | 27 | 20 | 0.554142 | 0.550279 |
| base_numeric | 22 | 16 | 0.319927 | 0.411565 |
| credit_mix_ohe | 3 | 2 | 0.072277 | 0.000973 |
| month_ohe | 9 | 1 | 0.033375 | 0.015789 |
| occupation_ohe | 15 | 0 | 0.006492 | 0.006686 |
| loan_type_ohe | 9 | 0 | 0.005817 | 0.005838 |
| payment_behaviour_ohe | 6 | 0 | 0.002694 | 0.002411 |
| payment_spend_level_ohe | 2 | 0 | 0.002598 | 0.001729 |
| payment_value_size_ohe | 3 | 0 | 0.001022 | 0.001439 |
| anomaly_indicators | 3 | 0 | 0.000827 | 0.001271 |
| missing_indicators | 5 | 0 | 0.000618 | 0.001536 |
| payment_min_amount_ohe | 3 | 0 | 0.000210 | 0.000485 |

## Файлы

- `lightgbm_feature_importance.csv` — все признаки с gain/split importance.
- `lightgbm_selected_features.csv` — выбранный набор признаков.
- `lightgbm_feature_family_importance.csv` — важность по группам признаков.
- `lightgbm_feature_selection_summary.json` — машинно-читаемое резюме.
