import time
from typing import Any, Callable, Sequence
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats import multivariate_normal as mul_norm


class Solution:
    '''求解所得信息的包装类.'''
    def __init__(self, success: bool, theta: Any, time: float, iter: int, msg: str) -> None:
        self.success = success
        self.theta = theta
        self.time = time
        self.iter = iter
        self.msg = msg

    def print(self) -> None:
        print(f'求解{"成功" if self.success else "失败"}！', )
        print(f'目标参数：{self.theta}')
        print(f'求解耗时：{self.time:.2f} s')
        print(f'迭代次数：{self.iter}')
        print(f'附加信息：{self.msg}')


class EM4GMM:
    '''EM算法针对GMM（高斯混合模型）优化后的类，只有一个静态方法可以调用.'''
    def __init__(self) -> None:
        ...

    @staticmethod
    def solve(
        dataset: Sequence[Sequence],
        init_pro: Sequence,
        init_mean: Sequence[Sequence],
        init_cov: Sequence[Sequence[Sequence]],
        iteration: int,
        tolerance: float = 1e-8,
    ) -> Solution:
        '''使用EM算法估计GMM的参数.

        Args:
            dataset: 原始数据集.
            init_pro: 初始参数值：各高斯分布的概率（标量）.
            init_mean: 初始参数值：各高斯分布的均值（向量）.
            init_cov: 初始参数值：各高斯分布的协方差（矩阵）.
            iteration: 最大迭代次数.
            tolerance: 参数优化容差，两次迭代间参数绝对变化量小于该容差时停止迭代.

        Returns:
            求解所得信息.
        '''
        t1 = time.time()

        K = len(init_pro)
        dataset = np.array(dataset)

        pro = [np.array(p) for p in init_pro]
        mean = [np.array(m) for m in init_mean]
        cov = [np.array(c) for c in init_cov]
        for j in range(iteration):
            if_break = True
            gamma = [pro[k] * mul_norm.pdf(dataset, mean=mean[k], cov=cov[k]) for k in range(K)]
            gamma_sum = np.sum(gamma, axis=0)
            gamma = [gamma[k] / gamma_sum for k in range(K)]
            for k in range(K):
                prev_pro, prev_mean, prev_cov = pro[k], mean[k], cov[k]
                mean[k] = np.sum(gamma[k].reshape(-1, 1) * dataset, axis=0) / np.sum(gamma[k])
                cov[k] = np.sum(
                    [g * (x - mean[k]).reshape(-1, 1) @ (x - mean[k]).reshape(1, -1) for g, x in zip(gamma[k], dataset)],
                    axis=0) / np.sum(gamma[k])
                pro[k] = np.mean(gamma[k])
                if (np.max([
                        np.abs(pro[k] - prev_pro),
                        np.max(np.abs(mean[k] - prev_mean)),
                        np.max(np.abs(cov[k] - prev_cov)),
                ]) > tolerance):  # 提前停止条件
                    if_break = False
            if if_break:
                break
        t2 = time.time()
        return Solution(True, {'pro': pro, 'mean': mean, 'cov': cov}, t2 - t1, j + 1, '')  # 返回解信息


if __name__ == "__main__":
    np.random.seed(0)  # 设定随机数种子

    data = pd.read_csv("height_data.csv")  # 读取数据

    dataset = np.array(data['height']).reshape(-1, 1)  # 同学身高数据集
    init_pro = [0.7, 0.3]
    init_mean = [[100], [120]]  # 先验：男生比女生高
    init_cov = [[[25]], [[25]]]
    iteration = 2000
    tolerance = 1e-8

    print('求解中...')

    solution = EM4GMM.solve(
        dataset,
        init_pro,
        init_mean,
        init_cov,
        iteration,
        tolerance,
    )

    print('求解完毕.')

    solution.print()

    # 根据条件概率计算某个样本属于哪一类
    theta = solution.theta
    pro_male = mul_norm.pdf(dataset, mean=theta['mean'][0], cov=theta['cov'][0])
    pro_female = mul_norm.pdf(dataset, mean=theta['mean'][1], cov=theta['cov'][1])
    genders = ['男' if x >= y else '女' for x, y in zip(pro_male, pro_female)]
    genders_real=['男' if x >= 500 else '女' for x in range(2000)]
    data['预测性别'] = genders
    data['实际性别'] = genders_real
    data.to_csv('height.csv', index=None)
    count=0
    print('性别判断结果：', genders)
    for i in range(2000):
        if genders[i]==genders_real[i]:
            count=count+1
    print('性别判断成功概率：', count/2000)

    theta = solution.theta
    plot_x = np.linspace(150, 200, 100)
    plot_ypm = norm.pdf(plot_x, loc=theta["mean"][0][0], scale=np.sqrt(theta["cov"][0][0, 0]))
    plot_ypf = norm.pdf(plot_x, loc=theta["mean"][1][0], scale=np.sqrt(theta["cov"][1][0, 0]))
    plot_yp = theta["pro"][0] * plot_ypm + theta["pro"][1] * plot_ypf

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.figure()
    plt.title('样本数据-预测概率密度函数')
    plt.ylabel('概率密度')
    plt.xlabel('身高/厘米')
    plt.plot(plot_x, plot_ypm)
    plt.plot(plot_x, plot_ypf)
    plt.plot(plot_x, plot_yp)
    plt.legend(['男生', '女生', '混合'])
    plt.show()
