# -*- coding: UTF-8 -*-

from perceptron import Perceptron

# defined the activate func


def f(x): return x


class LinearUnit(Perceptron):
    def __init__(self, input_num):
        '''init linear unit, set input param num'''
        # set the activate function with defined upon
        Perceptron.__init__(self, input_num, f)


def get_training_dataset():
    '''
    make salary of 5 man
    '''
    # make the training data
    # input a list of vector with every item be work time
    input_vecs = [[5], [3], [8], [1.4], [10.1]]
    # expected output vector, benefit one month, make sure one output to one input
    labels = [5500, 2300, 7600, 1800, 11400]
    return input_vecs, labels


def train_linear_unit():
    '''
    use the data to train linear unit
    '''
    # init perceptron with only one input feature (work time)
    lu = LinearUnit(1)
    # train, batch 10, rate 0.01
    input_vecs, labels = get_training_dataset()
    lu.train(input_vecs, labels, 10, 0.01)
    # return the unit
    return lu


def plot(linear_unit):
    import matplotlib.pyplot as plt
    input_vecs, labels = get_training_dataset()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter([x[0] for x in input_vecs], labels)
    weights = linear_unit.weights
    bias = linear_unit.bias
    x = range(0, 12, 1)
    y = [(weights[0] * item + bias) for item in x]
    ax.plot(x, y)
    plt.show()


if __name__ == '__main__':
    '''\u8bad\u7ec3\u7ebf\u6027\u5355\u5143'''
    linear_unit = train_linear_unit()
    # \u6253\u5370\u8bad\u7ec3\u83b7\u5f97\u7684\u6743\u91cd
    print(linear_unit)
    # \u6d4b\u8bd5
    print('Work 3.4 years, monthly salary = %.2f' % linear_unit.predict([3.4]))
    print('Work 15 years, monthly salary = %.2f' % linear_unit.predict([15]))
    print('Work 1.5 years, monthly salary = %.2f' % linear_unit.predict([1.5]))
    print('Work 6.3 years, monthly salary = %.2f' % linear_unit.predict([6.3]))
    plot(linear_unit)
