import scipy
import random
from scipy import constants
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fftpack import fft, fftshift
import matplotlib.pyplot as plt
import numpy as np
import math


class DataInput:
    def __init__(self, F_input):
        self.F_input = F_input

    F = dict(min=0, max=100, deg=10 ** (-9), err=0.1)
    F_min = 0
    F_max = 50
    F_deg = 10 ** (-9)
    F_err: float = 1 #шаг
    const = 18 **(5) * 927 * 10 **(-26)
    const /= 6.62 * 10 ** (-34)
    time_const = 2
    t = (scipy.constants.pi * 0.5) / (const * F_deg * (abs(F_max - F_min)))
    F_number = round((F_max - F_min + F_err) / F_err) - 1


def qubit(const, Field, t, deg):
    phi = const * Field * t * deg
    p0 = (math.sin(phi / 2)) ** 2
    return np.random.choice([0, 1], size=(1, 1), p=[p0, 1 - p0]).reshape(1)[0]


# Press the green button in the gutter to run the script.
def qubit_state(F, t, state):
    p1 = ((math.sin(data.const * F * data.F_deg * t / 2))*1) ** 2
    p0 = ((math.cos(data.const * F * data.F_deg * t / 2))*1) ** 2
    if state == 1:
        return p0
    else: return p1
def qubit_state1(F, t, state):
    p1 = ((math.sin(random.uniform(0, 1.5)))*1) ** 2
    p0 = ((math.cos(random.uniform(0, 1.5)))*1) ** 2
    if state == 1:
        return p0   
    else: return p1

if __name__ == '__main__':
    print("Enter field from 0 to 50:")
    f = int(input())
    data = DataInput(f)
    x0 = np.arange(data.F_min, data.F_max, data.F_err)
    y0 = np.full(len(x0), 1 / data.F_number)
    y10 = []
    attention = 0
    t = data.t
    print("Enter number of repeats:")
    k = int(input())
    p1 = ((math.sin(data.const * 50 * data.F_deg * t / 2)) * 1) ** 2
    p0 = ((math.cos(data.const * 1 * data.F_deg * t / 2)) * 1) ** 2
    #print(qubit_state(25*(10) **(-9), t, data.const))
    x0 = np.arange(data.F_min, data.F_max, data.F_err)
    for i in range(k):
        y = []
        y1 = []
        res = [0] * 1000
        for j in range(1000):
            res[j] = qubit(data.const, data.F_input, t, data.F_deg)
        if(res.count(0) >= res.count(1)):
            state = 0
        else:
            state = 1
        print(res.count(0), res.count(1))
        for q in range(50):
            y1.append(qubit_state((data.F_min + q * data.F_err), t, state))
            y.append(qubit_state((data.F_min + q * data.F_err), t, state) * y0[q])
        B = 0
        for j in range(len(y)):
            B += y[j]
        for q in range(len(y)):
            y[q] /= B
        y10 = y1
        y0 = y.copy()
        t *= 2
    fig, ax = plt.subplots()

    ax.plot(x0, y0, label=k)
    plt.plot(x0, y10)
    ax.set_xlabel('F')
    ax.set_ylabel('$P(F_i)$')
    ax.grid(True)
    ax.legend()
    plt.show()




"""

if __name__ == '__main__':
    data = DataInput(25)
    t = data.t
    k = int(input())
    x0 = np.arange(data.F_min, data.F_max, data.F_err)
    y0 = np.full(len(x0), 1 / data.F_number)
    y = y0
    print(data.F_number)
    for i in range(k):
        distribution = [0]* data.F_number
        for j in range(data.F_number):
            distribution[j] = qubit(data.const, data.F_input, t, data.F_deg)
            print(distribution[j])
        if( distribution.count(0) >= distribution.count(1)):
            state = 0
        else: state = 1
        y1 = [0]*data.F_number
        B = 0
        for j in range(data.F_number):
            B += y[j]* qubit_state((data.F_min + data.F_err * j)*data.F_deg, t, state)

        for q in range(data.F_number):
            y1[q] = y[q] * qubit_state((data.F_min + data.F_err * q)*data.F_deg, t, state)
            y1[q] /= B
        t = t * data.time_const
        y = y1
    plt.plot(x0, y0, label="first")
    plt.plot(x0, y, label="second")
    plt.grid(True)
    plt.legend()
    plt.show()



"""
"""

    for i in range(10):
        print(quibit(data.const, data.F_input, data.t, data.F_deg)) #записываю в массив, выбираю самый частый
    print(data.F_input, data.F_deg)
    data.F_distibution = quibit(data.const, data.F_input, data.t, data.F_deg) #определяю самый частый (0)

    x = np.arange(data.F_min, data.F_max, data.F_err)
    y = np.full(len(x), 1/data.F_number)
    distibution_1 = [0]*data.F_number
    for i in range(data.F_number):
        distibution_1[i] = quibit(data.const, data.F_input, t, data.F_deg)
    if (distibution_1.count(0) >= distibution_1.count(1)):
        state = 0
    else: state = 1
    y1 = []
    for i in range(data.F_number):
        y1[i] = y[i] * qubit_state((data.F_min + data.F_err * i)*data.F_deg, t, state)
    sum = 0
    for i in range(data.F_number):
        sum +=





   # plt.plot(x,y)

    y1 = np.full(len(x), 2 / data.F_number)
    plt.plot(x, y, label="first")
    plt.plot(x, y1, label="second")
    #plt.plot(x,y1)
    plt.grid(True)
    plt.legend()
    plt.show()
    for i in range(data.F_number):
        print(data.t)
    #заведеем цикл по F_number
    #массив плонтности вероятности функция принимает точку на отрезке и говорит насколько оно вероятно
#1/fnumber  - f_old
#sin-1 cos-0

"""
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
