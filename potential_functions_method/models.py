from typing import Callable


class AbstractModel:
    def learn(self, images, labels):
        raise NotImplementedError()

    def predict(self, image):
        raise NotImplementedError()

    def save(self, model_name=None):
        raise NotImplementedError()

    @classmethod
    def load(cls, model_name):
        raise NotImplementedError()


class ClassicModel(AbstractModel):
    def __init__(self, f: Callable, dimensions: int=2):
        """
        :param dimensions: размерность модели
        """
        self.f = f  # потенциальная функция
        self.history = []  # "история" для вычисления степени над e

        self.dimensions = dimensions

    @staticmethod
    def r(k, class_number):
        """ r """
        if class_number == 0:
            return 0 if k > 0 else 1
        elif class_number == 1:
            return 0 if k < 0 else -1
        else:
            raise Exception('Unknown class number, must be 0 or 1')

    def K(self, x):
        """ Значение кумулятивной функции """
        result = 0
        for idx, (r, xk) in enumerate(self.history):
            result += result + r * self.f(x, xk)

        return result

    def learn(self, images, labels, limit=100):
        """ Процесс обучения """

        assert len(images) == len(labels), 'Length of images and labels must be equal!'

        success = total = 0
        while total < limit:
            for xk_new, cls in zip(images, labels):
                total += 1
                r = self.r(self.K(xk_new), cls)
                if abs(r) == 1:
                    success = 0
                    # добавляем новую запись в полином над степенью e
                    self.history.append((r, xk_new))
                else:
                    success += 1
                    # если r равна 0 на протяжении N раз, то обучение завершено
                    if success >= len(images):
                        print(f'Break by success count. Total iterations: {total}, pow of e is: {len(self.history)}')
                        return total
        else:
            print(f'Stop by reaching limit {limit}')

        return total

    def predict(self, image, quiet=True):
        assert len(image) == self.dimensions, 'Dimensions of new image and model must be equal'

        res = self.K(image)
        if not quiet:
            print(f'K({image}): {res}')

        # номер класса, 0 или 1
        return int(res < 0)

    def save(self, model_name=None):
        pass

    @classmethod
    def load(cls, model_name):
        pass
