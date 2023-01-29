import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def Polynomial_fitting():
    x = [-40,20,60,120,200,400]
    x = np.array(x)
    print('x is :\n',x)
    num = [2.24,0.904,0.469,0.39,0.07,0.025]
    y = np.array(num)
    print('y is :\n',y)
    # f1 为各项的系数，3 表示想要拟合的最高次项是多少。
    f1 = np.polyfit(x, y, 4)
    # p1 为拟合的多项式表达式
    p1 = np.poly1d(f1)
    print('p1 is :\n',p1)
    plt.plot(x, y, 's',label='original values')
    yvals = p1(x) #拟合y值
    plt.plot(x, yvals, 'r',label='polyfit values')
    plt.show()

def curve_fitting():
    x = [-40,20,60,120,200,400]
    x = np.array(x)
    y = [2.24,0.904,0.469,0.39,0.07,0.025]
    y = np.array(y)

    # 这里的函数可以自定义任意形式。
    def func(x, a, b, c, d):
        return a + b*np.exp((x-c)/d)

    # popt返回的是给定模型的最优参数。我们可以使用pcov的值检测拟合的质量，其对角线元素值代表着每个参数的方差。
    popt, pcov = curve_fit(func, x, y)
    a = popt[0] 
    b = popt[1]
    c = popt[2]
    d = popt[3]
    yvals = func(x,a,b,c,d) #拟合y值
    plt.plot(x, y, 's',label='original values')
    plt.plot(x, yvals, 'r',label='polyfit values')
    plt.show()
    print(pcov)
    print('a:',a,'b:',b,"c:",c,"d",d)

if __name__ == '__main__':
    # curve_fitting()
    Polynomial_fitting()