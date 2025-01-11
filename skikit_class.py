from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from time import time


def load_data():
    """
        Загрузка датасета MNIST.
    - fetch_openml: загружаем данные датасета MNIST из онлайн хранилища.
    - Возвращаем данные (изображения) и метки (целевые значения).
    """
    dataset = fetch_openml('mnist_784', version=1)
    return dataset.data, dataset.target


def preprocess_data(x, y, test_size=0.2, random_state=42):
    """
        Разделение данных на обучающую и тестовую выборки. Нормализация.
    - test_size: указываем, какой процент данных будет использован для тестирования.
    - random_state: фиксируем случайность для воспроизводимости результатов.
    - scaler: создаем объект StandardScaler для нормализации данных.
    - fit_transform: нормализуем данные (вычисляет среднее и стандартное отклонение).
    - Возвращаем нормализованные обучающие и тестовые данные вместе с метками.
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)
    return x_train, x_test, y_train, y_test


def train_model(x_train, y_train, iteration):
    """
        Создание и обучение модели логистической регрессии.
    - max_iter: определяем максимальное количество итераций для сходимости алгоритма
                (максимальная точность была при 21й итерации (92.04%)).
    - model.fit: обучаем модель на обучающих данных.
    - Возвращаем обученную модель.
    """
    model = LogisticRegression(max_iter=iteration)
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """
        Оценка точности модели.
    - predict: используем модель для предсказания классов на тестовых данных.
    - accuracy: вычисляем точность предсказания, сравнивая предсказанные и истинные метки.
    - Выводим точность модели в процентах.
    """
    predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, predict)
    print(f'Анализ данных завершён.')
    print(f'Точность модели: {accuracy * 100:.2f}%')


def main():
    """
        Главная функция для запуска всех шагов.
    - iteration: максимальное число итераций, после которого алгоритм будет остановлен.
    - load_data: загружаем данные.
    - preprocess_data: предобрабатываем данные (разделяем и нормализуем).
    - train_model: обучаем модель на обучающих данных.
    - evaluate_model: оцениваем точность модели на тестовых данных.
    """
    iteration = 210
    
    start = time()  # Точка начала отсчета времени
    print(f'Начинаем анализ данных...')
    x, y = load_data()
    x_train, x_test, y_train, y_test = preprocess_data(x, y)
    model = train_model(x_train, y_train, iteration)
    evaluate_model(model, x_test, y_test)
    end = time()  # Точка окончания отсчета времени
    print(f'Время работы кода: {end - start} секунды.')


if __name__ == "__main__":  # Запуск основной функции
    main()
