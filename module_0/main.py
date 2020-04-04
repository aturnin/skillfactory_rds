import numpy as np

def game_core_v1(number, min_num, max_num):
    '''number   - загаднное число
       min_num  - минимальное возможное загаданное число
       max_num  - максимальное возможное загаднное число '''
    count = 0  
    while True:
        # начало попытки
        count += 1     
        # предсказание в центре диапазона возможных значений
        predict = int((min_num + max_num) * 0.5) 
        if number > predict: 
            min_num = predict + 1   # коррекция диапазона предсказаний по нижней границе
        elif number < predict: 
            max_num = predict - 1   # коррекция верхней границы  
        else: 
            return count # выход из цикла, если угадали
        
        
def score_game(game_core_v1):
    '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
    count_ls = []
    np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
    m1 = 1
    m2 = 100
    random_array = np.random.randint(m1, m2+1, size=(1000))
    for number in random_array:        
        count_ls.append(game_core_v1(number, m1, m2))
        
    score = int(np.mean(count_ls))
    print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")   
    return(score)

# запускаем
score_game(game_core_v1)