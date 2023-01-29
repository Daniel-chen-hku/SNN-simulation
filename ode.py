import matplotlib.pyplot as plt
from scipy import linspace,exp
from scipy.integrate import odeint, solve_bvp, solve_ivp
import numpy as np

'''
    为了兼容solve_ivp的参数形式，微分方程函数定义的参数顺序为(t,y)，因此使用odeint函数时需要使参数tfirst=True
    二阶甚至高阶微分方程组都可以变量替换成一阶方程组的形式，再调用相关函数进行求解，因此编写函数的时候，不同于一阶微分方程，二阶或者高阶微分方程返回的是低阶到高阶组成的方程组，

'''


def fvdp1(t,y):
    '''
    要把y看出一个向量，y = [dy0,dy1,dy2,...]分别表示y的n阶导，那么
    y[0]就是需要求解的函数，y[1]表示一阶导，y[2]表示二阶导，以此类推
    '''
    dy1 = y[1]      # y[1]=dy/dt，一阶导
    dy2 = -3 * y[1] - 2 * y[0] + exp( -1 * t ) 
    # y[0]是最初始，也就是需要求解的函数
    # 注意返回的顺序是[一阶导， 二阶导]，这就形成了一阶微分方程组
    return [dy1,dy2] 

# 或者下面写法更加简单
def fvdp2(t,y):
    '''
    要把y看出一个向量，y = [dy0,dy1,dy2,...]分别表示y的n阶导
    对于二阶微分方程，肯定是由0阶和1阶函数组合而成的，所以下面把y看成向量的话，y0表示最初始的函数，也就是我们要求解的函数，y1表示一阶导，对于高阶微分方程也可以以此类推
    '''
    y0, y1 = y   
    # y0是需要求解的函数，y1是一阶导
    # 返回的顺序是[一阶导， 二阶导]，这就形成了一阶微分方程组
    dydt = [y1, -3*y1-2*y0+exp(-t)] 
    
    return dydt

def solve_second_order_ode():
    '''
    求解二阶ODE
    '''
    t2 = linspace(0,20,1000)
    tspan = (0, 20.0)
    y0 = [1.0, 2.0] # 初值条件
    # 初值[2,0]表示y(0)=2,y'(0)=0
    # 返回y，其中y[:,0]是y[0]的值，就是最终解，y[:,1]是y'(x)的值
    y = odeint(fvdp1, y0, t2, tfirst=True)
    
    y_ = solve_ivp(fvdp2, t_span=tspan, y0=y0, t_eval=t2)
    
    plt.subplot(211)
    y1, = plt.plot(t2,y[:,0],label='y')
    y1_1, = plt.plot(t2,y[:,1],label='y‘')             
    plt.legend(handles=[y1,y1_1])
    
    plt.subplot(212)
    y2, = plt.plot(y_.t, y_.y[0,:],'g--',label='y(0)')
    y2_2, = plt.plot(y_.t, y_.y[1,:],'r-',label='y(1)')
    plt.legend(handles=[y2,y2_2])
    
    plt.show()
    
solve_second_order_ode()