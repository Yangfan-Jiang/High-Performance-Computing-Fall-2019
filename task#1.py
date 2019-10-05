# High Performance Computing task 1
import numpy as np
from matplotlib import pyplot as plt

def time_cost(n, p):
    return n**2/p + np.log2(p)
    
    
if __name__ == '__main__':
    n = [i*10 for i in range(1, 33)]
    p = [2**i for i in range(0, 8)]
    #print(n)
    #print(p)
    
    y = []
    x = []
    tmp_n = n[0]
    tmp_n = 50
    t0 = time_cost(tmp_n, 1)
    
    '''
    for tmp_p in p:
        x.append(tmp_p)
        y.append(time_cost(tmp_n, tmp_p))
        print(tmp_p)
        print('speed up', t0/y[-1])
        print('E       ', (t0/y[-1])/tmp_p)
        print()
    #plt.plot(x, y)
    #plt.show()
    '''
    tmp_p = 128
    for tmp_n in n:
        t0 = time_cost(tmp_n, 1)
        x.append(tmp_n)
        y.append(time_cost(tmp_n, tmp_p))
        print(tmp_n)
        print('speed up', t0/y[-1])
        print('E       ', (t0/y[-1])/tmp_p)
        print()
    #plt.plot(x, y)
    #plt.show()
    
