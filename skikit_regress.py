import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from time import time


def load_data(file_path):
    """
        Загрузка датасета из CSV файла.
    - read_csv: чтение данных CSV файла.
    - parse_dates: преобразование столбца 'Date' в формат даты 'datetime64'.
    - Возвращаем загруженный датасет.
    """
    dataset = pd.read_csv(file_path, parse_dates=['Date'])
    return dataset


def preprocess_data(data):
    """
        Предобработка данных: извлечение цен закрытия и дат.
    - price, date: извлекаем данные из столбцов 'Close', 'Date' (цена закрытия, дата)
                и преобразовываем их в двумерный массив (reshape(-1, 1)).
    - Возвращаем массивы цен и дат.
    """
    price = data['Close'].values.reshape(-1, 1)
    date = data['Date'].values.reshape(-1, 1)
    return price, date


def split_data(price, date, test_size=0.2):
    """
        Разделение данных на обучающую и тестовую выборки.
    - test_size: указываем, какой процент данных будет использован для тестирования.
    - shuffle: устанавливаем запрет на перемешивание данных (False)
    - Возвращаем выборки по цене (x) и по дате (y).
    """
    x_train, x_test = train_test_split(price, test_size=test_size, shuffle=False)
    y_train, y_test = train_test_split(date, test_size=test_size, shuffle=False)
    return x_train, x_test, y_train, y_test


def create_sequences(data, window, horizon):
    """
        Создание последовательностей данных для обучения модели.
    - x, y: инициализируем пустые списки для хранения последовательностей.
    - Проходим циклом по временным данным с учетом окна и горизонта прогнозирования.
    - window: количество дней, используемых в обучающей последовательности.
    - horizon: количество дней на которые делается прогноз.
    - Возвращаем массивы обучающих последовательностей (x) и целевых (y).
    """
    x = []
    y = []
    for i in range(len(data) - window - horizon):
        x.append(data[i:i + window])
        y.append(data[i + window + horizon - 1])
    return np.array(x), np.array(y)


def create_and_train_model(x_train, y_train):
    """
        Создаем и обучаем модель линейной регрессии.
    - LinearRegression: создаем экземпляр модели линейной регрессии.
    - fit: обучаем модель на обучающих данных.
    - Возвращаем обученную модель.
    """
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model


def make_predictions(model, x_test):
    """
        Производим предсказания на тестовых данных.
    - Возвращаем предсказанные данные.
    """
    return model.predict(x_test)


def plot_results(date, y_test, predict):
    """
        Визуализация фактических и предсказанных цен.
    - plot: строим график с фактическими значениями 'Actual'
                и предсказанными значениями 'Predicted'.
    """
    plt.plot(date[0:len(y_test)], y_test, label='Actual')
    plt.plot(date[0:len(predict)], predict, label='Predicted')
    plt.gcf().autofmt_xdate()  # Автоформатирование дат по оси X
    plt.title('SkiKit-Learn BTC Price Graph')  # Заголовок графика
    plt.xlabel('Date')  # Подпись по оси X
    plt.ylabel('USD')  # Подпись по оси Y
    plt.legend()  # Добавляем легенду
    plt.grid()  # Включаем сетку
    plt.show()  # Показываем график


def main():
    """
        Главная функция для запуска всех шагов.
    - window_size: определяем размеры окна данных для обучения.
    - horizon_size: определяем размеры горизонта прогнозирования.
    - load_data: загружаем данные из файла CSV.
    - preprocess_data: предобрабатываем данные для получения массивов цен и дат.
    - split_data: разделяем данные на обучающую и тестовую выборки.
    - create_sequences: создаем последовательности для обучающего (train) и тестового набора (test).
    - reshape: изменяем форму данных для обучения модели.
    - create_and_train_model: создаём модель и обучаем её на обучающих данных.
    - make_predictions: производим предсказания на тестовом наборе.
    - plot_results: визуализируем фактические и предсказанные результаты.
    """
    window_size = 30  # Число предыдущих дней для анализа
    horizon_size = 100  # Число дней для прогнозирования
    
    start = time()  # Точка начала отсчета времени
    print(f'Начинаем анализ данных...')
    dataset = load_data("BTC_data.csv")
    price, date = preprocess_data(dataset)
    train_price, test_price, train_date, test_date = split_data(price, date)
    x_train, y_train = create_sequences(train_price, window_size, horizon_size)
    x_test, y_test = create_sequences(test_price, window_size, horizon_size)
    x_train = x_train.reshape(-1, window_size)
    x_test = x_test.reshape(-1, window_size)
    model = create_and_train_model(x_train, y_train)
    prediction = make_predictions(model, x_test)
    end = time()  # Точка окончания отсчета времени
    print(f'Анализ данных завершён.')
    print(f'Время работы кода: {end - start} секунды.')
    plot_results(test_date[0:len(y_test)], y_test, prediction)


if __name__ == "__main__":  # Запуск основной функции
    main()
