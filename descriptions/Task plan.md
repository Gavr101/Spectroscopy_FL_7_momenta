ПЛАН ЭКСПЕРИМЕНТА (единый текст)

Эксперимент направлен на сравнение эффективности различных ортогональных базисов (Momentum, Legendre, Fourier, Zernike) для сжатия 2D-карт фотолюминесценции (ФЛ) и последующего предсказания концентраций ионов с помощью моделей машинного обучения.

1. Подготовка окружения

Импортируем все необходимые библиотеки:

numpy, pandas, matplotlib
sklearn (модели, метрики, CV, скейлеры)
xgboost
torch
json, tqdm
собственные трансформеры из src.py
2. Загрузка данных
Загружаем Y из Y_ions.csv
Загружаем X (карты ФЛ) через функцию import_df
Формируем массив X размера (N, H, W)
Проверяем согласованность индексов
3. Нормировка данных
Находим глобальный максимум по всем X
Делим все карты на этот максимум
Y нормируем с помощью MinMaxScaler:
scaler обучается на train
отдельно применяется к train и test
4. Определение трансформеров (базисов)

Создаем функцию-фабрику для:

MomentumTransformer2D
LegendreTransformer2D
FourierTransformer2D
ZernikeTransformer2D

Параметры:

варьируем order
x_bounds/y_bounds фиксированы (кроме Zernike — задаем явно)
5. Подготовка моделей
LinearRegression
RandomForestRegressor
XGBRegressor
MLPRegressor

Параметры:

единый validation split (где возможно)
фиксированные random_state
6. Реализация CNN-бейзлайна

Создаем класс CNNRegressor:

наследуем BaseEstimator, RegressorMixin
используем torch
реализуем:
init
fit (с early stopping)
predict
вход: (N, H, W) → reshape в (N, 1, H, W)
7. Реализация пайплайна

Для каждой комбинации:

модель
базис
порядок (order)

Шаги:

Разбиение на CV (KFold, 10)
Для каждого фолда:
split train/test
нормировка Y
применение трансформера (fit на train, transform на обоих)
обучение модели
предсказание
вычисление метрик:
MAE
RMSE
R2
8. Подсчет числа признаков
после transform получаем n_features
сохраняем как ключ
9. Структура результатов

Словарь вида:

results[model][momenta][n_features][metric] = [values per fold]
10. Бейзлайн CNN
обучаем аналогично CV
сохраняем метрики отдельно
считаем mean/std
11. Сохранение результатов
сохраняем основной словарь в JSON
сохраняем baseline отдельно
12. Анализ результатов

Для каждой модели:

7 подграфиков (по ионам)
X: n_features
Y: mean MAE
error bars: std
линии по базисам
baseline:
горизонтальная линия mean
полупрозрачная зона std
13. Визуализация
matplotlib
единые оси X
легенда по базисам