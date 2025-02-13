Дипломная работа по курсу Python-Разработчик.

  Тема: Сравнение различных библиотек для машинного обучения: scikit-learn, TensorFlow и PyTorch: 
Реализовать задачи классификации и регрессии с использованием scikit-learn, TensorFlow и PyTorch, 
сравнить их производительность и удобство использования.


Содержание:

1. Обзор проекта.
2. Выбор баз данных для обучения моделей.
3. Реализация задачи классификации путём Логистической Регрессии в SciKit-Learn.
4. Реализация задачи регрессии путём Линейной Регрессии в SciKit-Learn.
5. Реализация задачи классификации путём Логистической Регрессии в TensorFlow.
6. Реализация задачи регрессии путём Линейной Регрессии в TensorFlow.
7. Реализация задачи классификации путём Логистической Регрессии в PyTorch.
8. Реализация задачи регрессии путём Линейной Регрессии в PyTorch.
9. Заключение.
10. Список необходимых библиотек.

________________________________________________________________________________________________________________________________

1. Обзор проекта

	Проект подразумевает создание шести моделей нейронных сетей, для сравнения библиотек SciKit-Learn, TensorFlow и PyTorch 
при выполнять задачи классификации (с выводом точности в процентах) и регрессии (с построением графика).


2. Выбор баз данных для обучения моделей.

	Метод Классификации прекрасно подходит для узнавания объектов, по этому для обучения моделей была выбрана 
распространённая база данных образцов рукописного написания цифр MNIST. Она обладает большим объёмом данных для обучения 
и классов для предсказания (цифры от 0 до 9).
	Метод Регрессии прекрасно подходит для предсказания данных, по этому для обучения моделей был выбран 
набор статистических данных об изменениях курса биткоина за период с 2010 по 2024 год. В нем представлен 
широкий разброс значений и длительный период наблюдения.
	
![image](https://github.com/DrakeMk2/Urban-ML/blob/main/Image/mnist.jpg)


3. Реализация задачи классификации путём Логистической Регрессии в SciKit-Learn.

	Для реализации задачи был составлен код с использованием модели LogisticRegression и выводом результатов 
о точности узнавания в консоль. В ходе обучения 80% случайно выбранных данных были изпользованы для тренировки модели, а 20% данных 
для её последующего тестирования. В результате модель научилась узнавать цифры с точностью 91.61% (92.04% при 21й итерации). 
Для тренировки ей потребовалось около 30 секунд.

![image](https://github.com/DrakeMk2/Urban-ML/blob/main/Image/skikit_class.jpg)


4. Реализация задачи регрессии путём Линейной Регрессии в SciKit-Learn.

	Для реализации задачи был составлен код с использованием модели LinearRegression и выводом графика 
с реальными и предсказанными данными. В ходе обучения 80% линейно выбранных данных были изпользованы для тренировки модели, 
а 20% данных для её последующего тестирования. В результате модель научилась предсказывать дальнейшее движение курса валюты.
Для тренировки ей потребовалось меньше секунды.

![image](https://github.com/DrakeMk2/Urban-ML/blob/main/Image/skikit_regress.jpeg)


5. Реализация задачи классификации путём Логистической Регрессии в TensorFlow.

	Для реализации задачи был составлен код с использованием многослойной модели keras.Sequental и выводом результатов 
о точности узнавания в консоль. В ходе обучения 80% случайно выбранных данных были изпользованы для тренировки модели, а 20% данных 
для её последующего тестирования. В результате модель научилась узнавать цифры с точностью 95.65%. 
Для тренировки ей потребовалось около минуты.

![image](https://github.com/DrakeMk2/Urban-ML/blob/main/Image/tensor_class.jpg)


6. Реализация задачи регрессии путём Линейной Регрессии в TensorFlow.

	Для реализации задачи был составлен код с использованием многослойной модели keras.Sequental и выводом графика 
с реальными и предсказанными данными. В ходе обучения 80% линейно выбранных данных были изпользованы для тренировки модели, 
а 20% данных для её последующего тестирования. В результате модель научилась предсказывать дальнейшее движение курса валюты.
Для тренировки ей потребовалось около 35 секунд.

![image](https://github.com/DrakeMk2/Urban-ML/blob/main/Image/tensor_regress.jpeg)


7. Реализация задачи классификации путём Логистической Регрессии в PyTorch.

	Для реализации задачи был составлен код с использованием многослойной модели torch.nn.Sequential и выводом результатов 
о точности узнавания в консоль. В ходе обучения 80% случайно выбранных данных были изпользованы для тренировки модели, 
а 20% данных для её последующего тестирования. В результате модель научилась узнавать цифры с точностью 97.92%. 
Для тренировки ей требовалось чуть больше двух минут.

![image](https://github.com/DrakeMk2/Urban-ML/blob/main/Image/torch_class.jpg)


8. Реализация задачи регрессии путём Линейной Регрессии в PyTorch.

	Для реализации задачи был составлен код с использованием многослойной модели torch.nn.Sequential и выводом графика 
с реальными и предсказанными данными. В ходе обучения 80% линейно выбранных данных были изпользованы для тренировки модели, 
а 20% данных для её последующего тестирования. В результате модель научилась предсказывать дальнейшее движение курса валюты.
Для тренировки ей потребовалась одна секунда.

![image](https://github.com/DrakeMk2/Urban-ML/blob/main/Image/torch_regress.jpeg)


9. Заключение.

	Главное отличие трёх представленных библиотек в возможностях тонкой настройки моделей. 
- SciKit-Learn это фиксированные модули. Несёт в себе уже готовый набор инструментов не требующий сложных настроек. Заранее 
оптимизированные модели обрабатывают данные быстро, но из за невозможности повлиять на их структуру не очень точны. Ещё 
в SciKit-Learn присутствуют готовые модули для разделения и нормализации данных, которые удобно использовать в построении 
моделей на других библиотеках.
- TensorFlow это статический граф вычислений. На этапе описания возможно создать граф вычислений произвольного размера и сложности, 
однако после компиляции он станет монолитным. Этот подход комбинирует гибкость на этапе разработки и скорость в момент исполнения, 
но как и в функциональных языках, отладка становится весьма затруднительна. Данная библиотека показывает более высокую точность, 
но также растёт время обработки данных.
- PyTorch это динамический граф вычислений. Он слишком сложен для явного построения и существует лишь в в момент исполнения.
Граф строится динамически каждый раз при прямом проходе для того, чтобы затем иметь возможность сделать проход обратный. Подобный 
подход даёт максимальную гибкость и расширяемость, позволяет использовать в вычислениях все возможности используемого языка 
программирования и не ограничивает пользователя вообще ничем. Данная библиотека показывает самую высокую точность. Время 
обработки данных для задачи классификации увеличивается вдвое по сравнению с TensorFlow, но при выполнении задачи регрессии работает 
почти так же быстро как SciKit-Learn.

	В итоге мы видим что для простых задач прекрасно подходит SciKit-Learn. Золотой серединой из представленных библиотек 
является TensorFlow, сочитая в себе хорошие показатели точности, времени обработки данных и удобства в использовании. PyTorch 
в свою очередь обладает самыми глубокими возможностями в настройке моделей, что позволяет нам решать любые, даже весьма амбициозные 
задачи.


10. Список необходимых библиотек.

  https://github.com/DrakeMk2/Urban-ML/blob/main/requirements.txt
