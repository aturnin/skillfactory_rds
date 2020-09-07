# -*- coding: utf-8 -*-
'''
списки для фильтрации слов из отзывов и кухонь
'''

# список наиболее частых слов из отзывов
list_revword = ['good', 'food', 'best', 'not', 'place', 'worst', 'nice', 'great',
   'average', 'bad', 'amazing', 'poor', 'restaurant', 'terrible', 'excellent',
   'service', 'with', 'very', 'delicious', 'but', 'lunch', 'friendly',
   'ok', 'disappointing', 'experience', 'dinner', 'tasty', 'pizza',
   'you', 'on', 'coffee', 'breakfast', 'bar', 'fantastic', 'staff',
   'quality', 'italian', 'gem', 'local', 'meal', 'atmosphere', 'little',
   'dont', 'this', 'lovely', 'expensive', 'value', 'price', 'wonderful', 'cafe']

# список названий кухонь, которые больше влияют на успех модели (MAE = 0.1745)
cuis_list45 = ['Mediterranean', 'Central European', 'French',
   'Slovenian', 'Danish', 'Belgian', 'British', 'Pub', 'Czech', 'Hungarian',
   'Austrian', 'Vegan Options', 'Gluten Free Options', 'Portuguese', 'Scottish',
   'Polish', 'Spanish', 'Norwegian', 'German', 'Dutch', 'Pizza',
   'Scandinavian', 'Bar', 'Irish']

# список названий кухонь, которые лучше для разных разбиений
cuis_list = ['Vegan Options', 'European', 'Italian', 'Cafe', 'Mediterranean',
   'French', 'British', 'International', 'Fast Food', 'Gluten Free Options',
   'Pub', 'Vegetarian Friendly', 'Bar', 'Spanish', 'Wine Bar', 'American',
   'Seafood', 'Chinese', 'Asian', 'Pizza']