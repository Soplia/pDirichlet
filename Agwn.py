import torch
import numpy as np
import matplotlib.pyplot as plt

#wgn(x, snr)中x为信号，snr为信噪比
#10*log10( sum(x**2) / sum(n**2))
def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x**2)/len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)

#返回满足条件的高斯白噪声，只需要：
#x += wgn(x, snr)，即可以得到和matlab的awgn相同的效果
def awgn(x, snr):
    noise = wgn(x, snr)
    return x + noise

##hist()检查噪声是否是高斯分布
##psd()检查功率谱密度是否为常数
#t = np.arange(0, 1000000) * 0.1
#x = np.sin(t)
#n = wgn(x, 6)
#xn = x+n # 增加了6dBz信噪比噪声的信号
#plt.subplot(211)
#plt.hist(n, bins=100, normed=True)
#plt.subplot(212)
#plt.psd(n)
#plt.show()