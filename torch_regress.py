import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import Sequential, Flatten, Linear, ReLU, MSELoss
from torch import optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from time import time

scaler = StandardScaler()  # Создаем экземпляр стандартизатора


def load_data(file_path):
    """
        Загрузка датасета из CSV файла.
    - read_csv: чтение данных CSV файла.
    - parse_dates: преобразование столбца 'Date' в формат даты 'datetime64'.
    - Возвращаем загруженный датасет.
    """
    dataset = pd.read_csv(file_path, parse_dates=["Date"])
    return dataset


def preprocess_data(data):
    """
        Предобработка данных: извлечение цен закрытия и дат.
    - price, date: извлекаем данные из столбцов 'Close', 'Date' (цена закрытия, дата)
                и преобразовываем их в двумерный массив (reshape(-1, 1)).
    - fit_transform: стандартизируем цены.
    - Возвращаем массивы цен и дат.
    """
    price = data['Close'].values.reshape(-1, 1)
    date = data['Date'].values.reshape(-1, 1)
    price = scaler.fit_transform(price)
    return price, date


def split_data(price, date, test_size=0.2):
    """
        Разделение данных на обучающую и тестовую выборки.
    - test_size: указываем, какой процент данных будет использован для тестирования.
    - shuffle: устанавливаем запрет на перемешивание данных (False).
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


def create_model(window):
    """
        Создаем и возвращаем модель нейронной сети.
    - Flatten: преобразуем входные данные в двумерный массив.
    - Linear: полносвязные слои с активацией 'ReLU'.
                Выходной слой с одним нейроном для предсказания.
    - Возвращаем созданную модель.
    """
    model = Sequential(
        Flatten(),
        Linear(window, 64),
        ReLU(),
        Linear(64, 1),
        ReLU())
    return model


def train_model(model, x_train, y_train, criterion, optimizer, epochs):
    """
        Обучаем модель нейронной сети.
    - epochs: количество обучающих проходов модели.
    - model.train: устанавливаем режим обучения
    - optimizer.zero_grad: обнуляем градиенты перед новыми вычислениями.
    - outputs: прямой проход через модель.
    - loss: вычисляем значение потерь.
    - loss.backward: обратный проход для вычисления градиентов.
    - optimizer.step: обновляем параметры модели.
    - Выводим информацию о потере после каждые 10 эпох.
    - Возвращаем обученную модель.
    """
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(x_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')


def make_predictions(model, test):
    """
        Оцениваем модель на тестовой выборке и выводим её точность.
    - model.eval: переключаем модель в режим оценки (отключаем градиенты).
    - torch.no_grad: отключаем расчет градиентов.
    - Проходимся циклом по тестовым данным.
        - predict: получаем предсказания и преобразуем в numpy массив.
    - Возвращаем предсказания
    """
    model.eval()
    with torch.no_grad():
        predict = model(test).numpy()
        return predict


def plot_results(test_date, y_test, predict):
    """
        Визуализация фактических и предсказанных цен.
    - inverse_transform: обратное преобразование для фактических и предсказанных значений.
    - plot: строим график с фактическими значениями 'Actual'
                и предсказанными значениями 'Predicted'.
    """
    y_test = scaler.inverse_transform(y_test)
    predict = scaler.inverse_transform(predict)
    plt.plot(test_date[0:len(y_test)], y_test, label='Actual')
    plt.plot(test_date[0:len(predict)], predict, label='Predicted')
    plt.gcf().autofmt_xdate()  # Автоформатирование дат по оси X
    plt.title('PyTorch BTC Price Graph')  # Заголовок графика
    plt.xlabel('Date')  # Подпись по оси X
    plt.ylabel('USD')  # Подпись по оси Y
    plt.legend()  # Добавляем легенду
    plt.grid()  # Включаем сетку
    plt.show()  # Показываем график


def main():
    """
        Главная функция для запуска всех шагов.
    - epoch: количество обучающих проходов модели.
    - learning_rate: скорость обучения.
    - window_size: определяем размеры окна данных для обучения.
    - horizon_size: определяем размеры горизонта прогнозирования.
    - load_data: загружаем данные из файла CSV.
    - preprocess_data: предобрабатываем данные для получения массивов цен и дат.
    - split_data: разделяем данные на обучающую и тестовую выборки.
    - create_sequences: создаем последовательности для обучающего (train) и тестового набора (test).
    - torch.tensor: преобразуем данные в тензоры PyTorch.
    - create_model: создаём и компилируем модель оптимизатором SGD, с функцией среднеквадратичных потерь.
    - train_model: обучаем модель на обучающих данных.
    - make_predictions: производим предсказания на тестовом наборе.
    - plot_results: визуализируем фактические и предсказанные результаты.
    """
    epochs = 100
    learning_rate = 0.01
    window_size = 30  # Число предыдущих дней для анализа
    horizon_size = 100  # Число дней для прогнозирования
    
    start = time()  # Точка начала отсчета времени
    print(f'Начинаем анализ данных...')
    dataset = load_data('BTC_data.csv')
    price, date = preprocess_data(dataset)
    train_price, test_price, train_date, test_date = split_data(price, date)
    x_train, y_train = create_sequences(train_price, window_size, horizon_size)
    x_test, y_test = create_sequences(test_price, window_size, horizon_size)
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    model = create_model(window_size)
    criterion = MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_model(model, x_train, y_train, criterion, optimizer, epochs)
    predict = make_predictions(model, x_test)
    predict = torch.tensor(predict, dtype=torch.float32)
    end = time()  # Точка окончания отсчета времени
    print(f'Анализ данных завершён.')
    print(f'Время работы кода: {end - start} секунды.')
    plot_results(test_date[0:len(y_test)], y_test, predict)


if __name__ == "__main__":  # Запуск основной функции
    main()
