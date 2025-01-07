import torch
import torch.optim as optim
from torch.nn import Sequential, Flatten, Linear, ReLU, CrossEntropyLoss
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader


def load_data():
    """
        Загружаем датасет MNIST и нормализуем данные.
    - transform: преобразует изображения в тензоры.
    - MNIST: загружаем тренировочный и тестовый датасеты MNIST.
    - DataLoader: создаем загрузчики данных для тренировочного и тестового наборов с указанным размером партии.
    - Возвращаем нормализованные обучающие и тестовые данные вместе с метками.
    """
    transform = ToTensor()
    train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
    train_data = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_data = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)
    return train_data, test_data


def create_model():
    """
        Создаем и возвращаем модель нейронной сети.
    - Flatten: преобразуем входные данные в двумерный массив.
    - Linear: полносвязные слои с активацией 'ReLU'.
                Выходной слой с 10 нейронами (по количеству классов).
    - Возвращаем созданную модель.
    """
    model = Sequential(
        Flatten(),
        Linear(784, 128),
        ReLU(),
        Linear(128, 10))
    return model


def train_model(model, train_data, criterion, optimizer, epochs):
    """
        Обучаем модель нейронной сети.
    - loss: инициализируем переменную для потерь.
    - epochs: количество обучающих проходов модели.
    - model.train: устанавливаем режим обучения.
    - optimizer.zero_grad: обнуляем градиенты перед новыми вычислениями.
    - outputs: прямой проход через модель.
    - loss: вычисляем значение потерь.
    - loss.backward: обратный проход для вычисления градиентов.
    - optimizer.step: обновляем параметры модели.
    - Выводим информацию о потере после каждой эпохи.
    - Возвращаем обученную модель.
    """
    loss = 0
    for epoch in range(epochs):
        for images, labels in train_data:
            model.train()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')
    return model


def evaluate_model(model, test_data):
    """
        Оцениваем модель на тестовой выборке и выводим её точность.
    - model.eval: переключаем модель в режим оценки (отключаем градиенты).
    - total: счетчик всех тестовых примеров.
    - correct: счетчик правильных предсказаний.
    - torch.no_grad: отключаем расчет градиентов.
    - Проходимся циклом по тестовым данным.
        - outputs: получаем предсказания от модели.
        - predicted: получаем индекс класса с максимальной вероятностью.
        - total: увеличиваем общий счетчик.
        - correct: увеличиваем счетчик правильных предсказаний.
    - accuracy: вычисляем точность модели.
    - print(accuracy): выводим точность модели в процентах.
    """
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_data:
            outputs = model(images)
            _, predict = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Точность модели: {accuracy:.2f}%')


def main():
    """
        Главная функция для запуска всех шагов.
    - epoch: количество обучающих проходов модели.
    - learning_rate: скорость обучения.
    - load_data: загружаем и нормализуем данные и метки.
    - create_model: создаем модель.
    - criterion: определяем функцию потерь (кросс-энтропия).
    - optimizer: оптимизатор SGD с заданной скоростью обучения.
    - train_model: обучаем модель заданное количество эпох.
    - evaluate_model: оцениваем модель на тестовых данных.
    """
    epochs = 10
    learning_rate = 0.1

    train_loader, test_loader = load_data()
    model = create_model()
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    train_model(model, train_loader, criterion, optimizer, epochs)
    evaluate_model(model, test_loader)


if __name__ == "__main__":  # Запуск основной функции
    main()
