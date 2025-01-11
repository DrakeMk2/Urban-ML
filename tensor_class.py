from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from tensorflow.keras.utils import to_categorical
from time import time
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Устанавливаем уровень логирования (для скрытия предупреждений о моём CPU)


def load_data():
    """
        Загружаем датасет MNIST и нормализуем данные.
    - fetch_openml: загружаем датасет MNIST из OpenML.
    - data / 255.0: приводим значения пикселей к диапазону от 0 до 1.
    - target.astype: преобразуем метки в формат float32 для дальнейшей обработки.
    - Возвращаем нормализованные данные (x) и метки (y).
    """
    dataset = fetch_openml('mnist_784', version=1)
    x = dataset.data / 255.0
    y = dataset.target.astype('float32')
    return x, y


def split_data(x, y):
    """
        Разделяем данные на обучающую и тестовую выборки.
    - test_size: указываем, какой процент данных будет использован для тестирования.
    - random_state: фиксируем случайность для воспроизводимости результатов.
    - Возвращаем разделенные выборки.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test


#
def preprocess_data(y_train, y_test):
    """
        Предобработка меток.
    - to_categorical: преобразуем метки обучающей (train) и тестовой (test) выборки в категориальный формат.
    - num_classes: общее количество классов что мы хотим получить (10: числа от 0 до 9).
    - Возвращаем предобработанные метки.
    """
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return y_train, y_test


def create_model():
    """
        Создаем и возвращаем модель нейронной сети.
    - Flatten: преобразуем входные данные 28x28 в одномерный массив 784.
    - Dense: полносвязные слои с активацией 'relu' и 'softmax'.
                Выходной слой с 10 нейронами (по количеству классов).
    - Возвращаем созданную модель.
    """
    model = Sequential([
        Flatten(input_shape=(784,)),
        Dense(128, activation='relu'),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')])
    return model


def compile_and_train_model(model, x_train, y_train, epochs, optimizer):
    """
        Компилируем и обучаем модель нейронной сети.
    - model.compile: компилируем модель оптимизатором Adam,
                с функцией потерь категориальной кросс-энтропии и метрикой точности.
    - model.fit: обучаем модель в 10 эпох, с размером партии 32, используя 20% данных для валидации.
    - Возвращаем обученную модель.
    """
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2)
    return model


def evaluate_model(model, x_test, y_test):
    """
        Оцениваем модель на тестовой выборке и выводим её точность.
    - model.evaluate: оцениваем модель на тестовых данных.
    - print(accuracy): выводим точность модели в процентах.
    """
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f'Анализ данных завершён.')
    print(f'Точность модели: {accuracy * 100:.2f}%')


def main():
    """
        Главная функция для запуска всех шагов.
    - epoch: количество обучающих проходов модели.
    - learning_rate: скорость обучения.
    - load_data: загружаем данные и метки.
    - split_data: разделяем данные на обучающую и тестовую выборки.
    - preprocess_data: предобрабатываем метки (разделяем и категорируем).
    - create_model: создаем модель.
    - compile_and_train_model: компилируем и обучаем модель.
    - evaluate_model: оцениваем модель на тестовой выборке и выводим результат.
    """
    epochs = 10
    learning_rate = 0.01
    
    start = time()  # Точка начала отсчета времени
    print(f'Начинаем анализ данных...')
    x, y = load_data()
    x_train, x_test, y_train, y_test = split_data(x, y)
    y_train, y_test = preprocess_data(y_train, y_test)
    model = create_model()
    optimizer = Adam(learning_rate=learning_rate)
    compile_and_train_model(model, x_train, y_train, epochs, optimizer)
    evaluate_model(model, x_test, y_test)
    end = time()  # Точка окончания отсчета времени
    print(f'Время работы кода: {end - start} секунды.')


if __name__ == "__main__":  # Запуск основной функции
    main()
