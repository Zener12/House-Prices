# House Prices — Advanced Regression Techniques

Kaggle competition: предсказание цен на жильё по 79 признакам.  
Метрика: RMSLE (Root Mean Squared Log Error).

## Результаты

| # | Изменение | Kaggle Score | Место |
|---|-----------|:------------:|:-----:|
| 1 | CatBoost baseline | 0.12894 | 1617 / 5015 |
| 2 | Обучение на всём трейне | 0.12768 | 1401 / 5015 |
| 3 | CatBoost + CV in Optuna | 0.12768 | 1405 / 5039 |
| 4 | **Ансамбль 0.8×CB + 0.2×LGBM** | **0.12362** | **817 / 5042** |
| 5 | CB + EN + 5 признаков (NaN баг) | 0.15062 | — |
| 6 | CB + EN, bugfix | 0.15062 | — |
| 7 | Только CatBoost (Optuna 5-fold) | 0.12409 | — |
| 8 | Только CatBoost (Optuna 3-fold) | 0.12453 | — |

Лучший результат: **0.12362**, топ ~16% (817 / 5042).

## Структура проекта

```
House Prices/
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── house_price.ipynb   # основной ноутбук
├── cb_best_params.json # сохранённые параметры CatBoost (Optuna)
├── submission.csv      # последний сабмит
└── requirements.txt
```

## Методология

### 1. EDA
- Анализ распределений числовых и категориальных признаков
- Корреляционная матрица топ-15 фичей
- Удаление двух аномальных домов (огромная площадь при низкой цене)
- Логарифмирование таргета (SalePrice) — требование метрики RMSLE

### 2. Feature Engineering & Preprocessing
**Импутация:** медиана для числовых, мода/категория "None" для категориальных, 0 для GarageYrBlt (нет гаража).

**Инженерные признаки (базовые):**
- `TotalSF` = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
- `HouseAge`, `RemodAge` — возраст дома и время с ремонта
- `HasPool` — бинарный признак наличия бассейна

**Инженерные признаки (SHAP-driven, добавлены после анализа):**
- `QualSF` = OverallQual × TotalSF — стал признаком №1 по важности
- `TotalBath` — суммарные санузлы с логичными весами (full=1, half=0.5)
- `TotalPorch` — суммарная площадь внешних зон
- `IsNew` — новостройка в год продажи
- `OverallScore` = OverallQual × OverallCond

**Кодирование категорий (для LightGBM/ElasticNet):**
- Ordinal Encoding — признаки с естественным порядком (Qual/Cond/etc.)
- OHE — признаки с малой кардинальностью (< 10 категорий)
- Target Encoding с out-of-fold защитой от утечки — высококардинальные (Neighborhood, Exterior и др.)

CatBoost работает с категориями нативно — отдельный датафрейм без OHE/TE.

### 3. Модели

**CatBoost** (основная модель):
- Optuna (50 триалов, 3-fold CV) → лучший CV RMSE: 0.11577
- Лучшие параметры: lr≈0.022, max_depth=5, l2_leaf_reg≈1.08
- Обучение с early stopping (50 итераций)

**LightGBM:**
- Optuna (50 триалов, 5-fold CV)
- Val RMSE ≈ 0.1327 (уступает CatBoost на этом датасете из-за нативной обработки категорий)

**ElasticNet:**
- ElasticNetCV с 5-fold CV для выбора alpha и l1_ratio, RobustScaler
- Val RMSE ≈ 0.130, но сжимает предсказания на тесте → ухудшает финальный score

### 4. Ансамбль
Перебор весов на валидационной выборке. Лучший результат — 0.8×CatBoost + 0.2×LightGBM (submission #4). ElasticNet в ансамбль не пошёл: высокая корреляция ошибок с бустингами + недисперсные предсказания.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск

Открыть `house_price.ipynb` и запустить ячейки последовательно. Параметры Optuna кэшируются в `cb_best_params.json` — повторный запуск не пересчитывает.
