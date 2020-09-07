# -*- coding: utf-8 -*-
"""
Created on Sat Sep  5 17:47:16 2020

@author: alex
"""
# функции проекта

# import numpy as np # linear algebra
import pandas as pd
import numpy as np
import collections as coll
from datetime import datetime as dtm
import re
from sklearn.preprocessing import MultiLabelBinarizer
import list_param as lp


def preproc_data(df_input):
    '''includes several functions to pre-process the predictor data.'''

    df0 = df_input.copy()

    # %% ################### 1. Предобработка ##############################################################
    # убираем не нужные для модели признаки
    df0.columns = ['rest_id', 'city', 'cuisine', 'ranking',
               'price_range', 'num_reviews', 'reviews', 'url_ta', 'id_ta', 'sample', 'rating']
    df0.drop(['rest_id', 'url_ta', 'id_ta'], axis=1, inplace=True)


    # %% ################### 2. NAN ##############################################################
    # Далее заполняем пропуски, вы можете попробовать заполнением средним или средним по городу и тд...
    # df0['Number of Reviews'].fillna(0, inplace=True)
    # тут ваш код по обработке NAN

    # кухни преобразовать в список
    df0.cuisine = df0.cuisine.apply(checkstr_nan)
    df0.cuisine = df0.cuisine.apply(split_cuisine)

    # частоты кухонь по городам
    city_cuis_m, city_cuis_f = freq_cuis(df0)

    # set_cuisine = set()
    # for city1, l_cui1 in city_cuis_f.items():
    #     for cui1 in l_cui1:
    #         set_cuisine.add(cui1)

    # заполним пропущенные кухни наиболее частыми для этого города
    df0 = df0.apply(lambda x: fill_cuis(x, city_cuis_m), axis=1)

    df0['cuisine'].fillna('None', inplace=True)

    # заполнить пустые цены и заменить цифрами, заполнить пропущенные
    # наиболее частой ценой (2) плохо
    # 0! - так лучше
    df0.price_range = df0.price_range.apply(fill_price)

    # пустые num_reviews заменить средним значением - тоже плохо
    # mean_numrev = int(df0[df0.num_reviews.notna()].num_reviews.median())
    mean_numrev = 0
    # пустые num_reviews заменить средним значением
    df0['num_reviews'].fillna(mean_numrev, inplace=True)

    # %% разбираем отзывы
    df_rev = pd.DataFrame(df0.reviews.apply(split_reviews))
    df_rev.columns = ['reviews']
    # для разделённых дат и отзывов
    df_wrd, df_dt = parse_reviews(df_rev)

    # %% ################### 3. Encoding ##############################################################

    # города
    df0 = pd.get_dummies(df0, columns=[ 'city',]) # , dummy_na=True
    mlb = MultiLabelBinarizer()

    # кухни
    dfdm_cuis =  pd.DataFrame(mlb.fit_transform(df0.cuisine),columns=mlb.classes_, index=df0.index)
    object_columns = ['cuisine', 'reviews' ]
    df0.drop(object_columns, axis = 1, inplace=True)
    df0 = pd.concat([df0, dfdm_cuis[lp.cuis_list]], axis=1)

    # отзывы и даты
    dfdm_rev = pd.DataFrame(mlb.fit_transform(df_wrd.words), columns=mlb.classes_, index=df_wrd.index)
    dfdm_date = pd.DataFrame(mlb.fit_transform(df_dt.date),columns=mlb.classes_, index=df_dt.index)
    df0 = concate_3df(df0, dfdm_rev, dfdm_date)

    return df0



def checkstr_nan(row):
    # заменить nan на None
    if type(row) == str:
        return row
    else:
        return None


def split_cuisine(x):
    # получить список кухонь из строки
    if not ( x == None ):
        lst = x.strip('][').split(', ')
        res = list()
        for c in lst:
            res.append((c.strip('\'')))
        return res
    else:
        return None


# %% заполнение кухнями (частыми в данном городе)
# строк с пустыми значениями

def freq_cuis(df0):
    # города в список
    l_city = list(set(df0.city))

    # кухни по городам
    city_cuis = dict()

    # среднее число кухонь в ресторане по городам
    mean_cuis = dict()

    for city1 in l_city:        # для каждого города
        cc = coll.Counter()     # счётчик каждой кухни в заданном городе
        city_sel = df0[df0.city == city1]   # данные для одного города
        m_c = list()                        # список для числа кухонь в строке
        for cuis_l in city_sel.cuisine:     # из строки берём список кухонь
            if not (cuis_l== None):
                m_c.append(len(cuis_l))     # добавляем число кухонь
                for cuis1 in cuis_l:
                    cc[cuis1] +=1       # считаем кухни

        # в каждом городе
        city_cuis[city1] = cc.most_common()             # счётчик по всем кухням
        mean_cuis[city1] = int(np.ceil(np.mean(m_c)))   # среднее число кухонь

    # частоты кухонь по городам
    city_cuis_m = dict()
    city_cuis_f = dict() # полный список

    for ct, cui in city_cuis.items():
        m = mean_cuis[ct]   # среднее число кухонь в городе
        l_cui = list()
        f_cui = list()
        for i in range(m):
            l_cui.append(cui[0:m][i][0]) # названия частых кухонь в список

        n = len(cui[0:-1][i][0])
        for i in range(n):
            f_cui.append(cui[0:-1][i][0]) # названия частых кухонь в список


        city_cuis_m[ct] = l_cui
        city_cuis_f[ct] = f_cui

    return city_cuis_m, city_cuis_f


def fill_cuis(restoran, city_cuis_m):
    if restoran.cuisine == None:
        restoran.cuisine = city_cuis_m[restoran.city]
    return restoran


def fill_price(row):
    if type(row) == str:
        if len(row) == 1:
            return 1
        elif len(row) == 4:
            return 3
        else:
            return 2
    else:
        return 0 # если цен нет (то, самая низкая)


def fill_numrev(row, mean_numrev):
    if np.isnan(row):
        return mean_numrev
    else:
        return row


def split_reviews(x):
    if type(x) == float:
        x = '[[],[]]'
    res = list()
    x0 = x.strip('][').split('], [')
    for rx in x0:
        otz = rx.split('\', ')
        for x1 in otz:
            x1 = x1.strip('\'')
            res.append(x1)

    return res


def concate_3df(df1, df2, df3):

    np_1 = df1.to_numpy()
    np_2 = df2.to_numpy()
    np_3 = df3.to_numpy()

    np_1 = np.concatenate((np_1,np_2),axis=1)
    np_1 = np.concatenate((np_1,np_3),axis=1)

    np_col = np.concatenate((df1.columns, df2.columns, df3.columns))

    return pd.DataFrame(np_1, columns=np_col)


def parse_reviews(df_rev):
    '''
    Parameters
    ----------
    df_rev : DataFrame
        DESCRIPTION.
        DataFrame со строками отзывов и датами

    Returns
    -------
    df_wrd : DataFrame
        DESCRIPTION.
        DataFrame со списком слов из отзыва (самых часты по базе)

    df_dt : DataFrame
        DESCRIPTION.
        DataFrame со списком дат из отзыва.
    '''

    # списки для списков слов из отзывов в каждой строке
    l_wrd = list()
    # списки для списков дат ...
    l_dt = list()

    # счётчики для частоты слов и дат
    c_word = coll.Counter()
    c_date = coll.Counter()

    # число слов в отзыве
    len_rev = list()

    for row in df_rev.reviews:

        # списки для дат и слов из отзыва
        dt_list = list()
        wrd_list = list()

        for cell1 in row:
            cell2 = re.search('\d+/\d+/\d+', cell1) # если дата
            if cell2:
                str_dt = cell2.group()
                num_all = re.findall('\d+', str_dt)
                if int(num_all[0]) <= 12:
                    if int(num_all[2]) > 100:
                        dt_1 = dtm.strptime(str_dt, '%m/%d/%Y')
                    else:
                        dt_1 = dtm.strptime(str_dt, '%m/%d/%y')
                else:
                    if int(num_all[2]) > 100:
                        dt_1 = dtm.strptime(str_dt, '%d/%m/%Y')
                    else:
                        dt_1 = dtm.strptime(str_dt, '%d/%m/%y')

                c_date[dt_1] += 1

                # добавляем признаки, которые улучшают модель
                dt_list.append(dt_1.year) # month weekday()
                dt_list.append(dt_1.weekday())

            else:
                cell2 = cell1.lower()
                cell2 = re.sub('[^\s\d\w]', '', cell2)
                l_cell = re.split('\s+', cell2)
                len_rev.append(len(l_cell))
                for l1 in l_cell:
                    c_word[l1] += 1

                for l1 in l_cell:
                    if len(l1) != 0:
                        wrd_list.append(l1)

        if len(dt_list) == 0:
            dt_list.append(np.nan)


        if len(wrd_list) == 0:
            wrd_list = ['None']

        wrd_list_sel = list()
        for w1 in wrd_list:
            if w1 in lp.list_revword:
                wrd_list_sel.append(w1)

        if len(wrd_list_sel) == 0:
           wrd_list_sel = ['None']

        l_wrd.append(wrd_list_sel)
        l_dt.append(dt_list)

    # создаём датафреймы для вывода
    df_wrd = pd.DataFrame(data=np.array(l_wrd, dtype=object), columns=['words'])
    df_dt = pd.DataFrame(data=np.array(l_dt, dtype=object), columns=['date'])

    # анализ частоты встречающихся в отзывах дат
    df_countd = pd.DataFrame(c_date.most_common())
    df_countd.columns = ['date', 'num_date']
    df_countd = df_countd.sort_values(by='date', ignore_index=True)

    # fig = plt.figure(1)
    # axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    # axes.scatter(x = df_countd['date'], y = df_countd['num_date'], color='red')

    # fig = plt.figure(2)
    # axes = fig.add_axes([0.1, 0.1, 0.9, 0.9])
    # axes.scatter(x = df_countd['date'].dt.month, y = df_countd['num_date'], color='red')


    # далее средние данные по дате и словам отзыва, которые не нужны,
    # модель работает лучше с признаком пропуска
    df_cwrd = pd.DataFrame(c_word.most_common())
    df_cwrd.columns = ['word', 'num_word']
    # самые частые 200 слов в отзывах
    l_wordrev = list(df_cwrd.word[1:200])

    # число слов в отзыве
    mean_rev = np.mean(len_rev)
    print('Среднее число слов в отзыве ', int(np.ceil(mean_rev)))

    # самые частые слова в отзывах (для заполнения, отсутствующих отзывов)
    freq_rev = ['food', 'good', 'great', 'nice']

    # частая дата
    df_countd.num_date.max()
    freq_date = df_countd.loc[df_countd.num_date.idxmax()][0]
    print('Freq date = ', freq_date)

    return df_wrd, df_dt