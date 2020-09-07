# -*- coding: utf-8 -*-
# проект о вкусной и здоровой пище

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

RANDOM_SEED = 1
DATA_DIR = 'kaggle/input/sf-dst-restaurant-rating'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
# df_train.info()

# Объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями
data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
# data.info()

# Для примера я возьму столбец Number of Reviews
data['Number_of_Reviews_isNAN'] = pd.isna(data['Number of Reviews']).astype('uint8')
# data['Number_of_Reviews_isNAN']
# Далее заполняем пропуски 0, вы можете попробовать заполнением средним или средним по городу и тд...
data['Number of Reviews'].fillna(0, inplace=True)

# %% Обработка признаков
data.nunique(dropna=False)

# для One-Hot Encoding в pandas есть готовая функция - get_dummies. Особенно радует параметр dummy_na
data = pd.get_dummies(data, columns=[ 'City',], dummy_na=True)
# data.head(5)

data['Price Range'].value_counts()

# Ваша обработка 'Price Range'
# тут ваш код на обработку других признаков
# .....

# plt.figure(1)
# plt.rcParams['figure.figsize'] = (10,7)
# df_train['Ranking'].hist(bins=100)

# plt.figure(2)
# df_train['City'].value_counts(ascending=True).plot(kind='barh')

# plt.figure(3)
# df_train['Ranking'][df_train['City'] =='London'].hist(bins=100)


# # посмотрим на топ 10 городов
# plt.figure(4)
# for x in (df_train['City'].value_counts())[0:10].index:
#     df_train['Ranking'][df_train['City'] == x].hist(bins=100)
# plt.show()


# целевая переменная
# plt.figure(5)
# df_train['Rating'].value_counts(ascending=True).plot(kind='barh')

# # целевая переменная относительно признака
# plt.figure(6)
# df_train['Ranking'][df_train['Rating'] == 5].hist(bins=100)

# plt.figure(7)
# df_train['Ranking'][df_train['Rating'] < 4].hist(bins=100)

# # корреляция признаков

# plt.rcParams['figure.figsize'] = (15,10)
# sns.heatmap(data.drop(['sample'], axis=1).corr(),)



# %% Data Preprocessing
# Теперь, для удобства и воспроизводимости кода, завернем всю обработку в одну большую функцию.


# на всякий случай, заново подгружаем данные
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'/kaggle_task.csv')
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем
# data.info()

from func_v2 import preproc_data
df_preproc = preproc_data(data)
# df_preproc.sample(10)
# df_preproc.info()

# %% Теперь выделим тестовую часть

train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

y = train_data.rating.values            # наш таргет
X = train_data.drop(['rating'], axis=1)

# Воспользуемся специальной функцие train_test_split для разбивки тестовых данных
# выделим 20% данных на валидацию (параметр test_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)

# проверяем
test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape

# %% Модель

# Импортируем необходимые библиотеки:
from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics # инструменты для оценки точности модели

# Создаём модель (НАСТРОЙКИ НЕ ТРОГАЕМ)
model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)
y_pred = np.round(y_pred * 2) / 2 # оценки кратны 0.5

# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.
print('MAE:', metrics.mean_absolute_error(y_test, y_pred))


# %% в RandomForestRegressor есть возможность вывести самые важные признаки для модели
plt.figure(8)
plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(50).plot(kind='barh')

# %% на каггл
# test_data.sample(10)
test_data = test_data.drop(['rating'], axis=1)
# sample_submission

predict_submission = model.predict(test_data)
predict_submission = np.round(predict_submission * 2) / 2 # оценки кратны 0.5
# predict_submission

sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
# sample_submission.head(10)



