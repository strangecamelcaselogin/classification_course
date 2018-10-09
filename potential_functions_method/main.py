from pathlib import Path
import math
import matplotlib.pyplot as plt

from potential_functions_method.models import ClassicModel, AbstractModel, StochasticModel


def sq_dist(x, y):
    """ Квадрат дистанции """
    return sum((xi - yi)**2 for xi, yi in zip(x, y))

def dist(x, y):
    """ Дистанция """
    return math.sqrt(sum((xi - yi)**2 for xi, yi in zip(x, y)))


def viz2d(model: AbstractModel, train_data, border=(-25, 25, 25, -25), step:int = 1):
    """ Визуализация классов в пространстве признаков """
    x1, y1, x2, y2 = border

    # точечный график классов в пространстве признаков
    cls_values = [[[], []], [[], []]]
    for x in range(x1, x2, step):
        for y in range(y1, y2, -step):
            cls = model.predict((x, y))
            cls_values[cls][0].append(x)
            cls_values[cls][1].append(y)

    # plt.axhline(y=0, color='k')
    # plt.axvline(x=0, color='k')
    for cls, (x, y) in enumerate(cls_values):
        plt.plot(x, y, 'o', alpha=0.9)

    plt.legend([f'class #{i+1}' for i in range(2)])
    plt.show()

    # точки и их классы из обучающей выборки
    points, labels = train_data
    cls_train = [[[], []], [[], []]]
    for p, cls in zip(points, labels):
        cls_train[cls][0].append(p[0])
        cls_train[cls][1].append(p[1])

    plt.grid()
    # plt.axhline(y=0, color='k')
    # plt.axvline(x=0, color='k')
    for cls, (x, y) in enumerate(cls_train):
        plt.plot(x, y, 'o')

    plt.legend([f'class #{i+1}' for i in range(2)])
    plt.show()


def parse_line(line):
    return tuple(map(float, line.strip().split(' ')))


def load_data(name, directory):
    with open(directory / Path(name)) as data_file:
        dimentions = int(data_file.readline())
        images, labels = [], []
        for line in data_file:
            try:
                cls, *img = parse_line(line)
                images.append(img)
                labels.append(int(cls))
            except (ValueError, TypeError) as e:
                raise Exception('Dataset error') from e

        return dimentions, images, labels


def f(x, xk, a=1):
    """ Потенциальная функция: K(x, xk) """
    return a * math.e ** (-a * sq_dist(x, xk))


def interact(model):
    try:
        while True:
            i = input('Образ (через пробел): ')

            if i.lower() in ['e', 'exit']:
                return

            try:
                i = parse_line(i)
            except (ValueError, TypeError) as e:
                print('Неверный входной формат образа.\n')
                continue

            try:
                c = model.predict(i)
                print(f'Класс: #{c + 1}\n')
            except AssertionError as e:
                print(f'При вычислении возникла ошибка.\n  "{e}"')

    except KeyboardInterrupt:
        print('Выход')


def run_classic(d, images, labels):
    model = ClassicModel(f, dimensions=d)

    # произведем обучение классификатора
    model.learn(images, labels)

    if d == 2:
        viz2d(model, (images, labels))

    # m.save()  # todo дать имя модели

    interact(model)


def run_stochastic(d, images, labels):
    model = StochasticModel(f, dimensions=d)

    model.learn(images, labels)

    if d == 2:
        viz2d(model, (images, labels))

    # m.save()  # todo дать имя модели

    interact(model)


if __name__ == '__main__':
    data = Path('./data')

    # 'simple1d.txt'
    # 'simple2d.txt', 'advanced2d.txt'
    data_names = ['simple1d.txt']

    for d_name in data_names:
        d, images, labels = load_data(d_name, data)
        # run_classic(d, images, labels)
        run_stochastic(d, images, labels)
