import geatpy as ea  # import geatpy
import numpy as np

if __name__ == '__main__':
    # 问题对象
    problem = ea.benchmarks.Ackley(10)
    # 构建算法
    algorithm = ea.soea_SEGA_templet(
        problem,
        ea.Population(Encoding='RI', NIND=100),
        MAXGEN=100,  # 最大进化代数。
        logTras=1)  # 表示每隔多少代记录一次日志信息，0表示不记录。
    # algorithm.mutOper.F = 0.5  # 差分进化中的参数F
    # algorithm.recOper.XOVR = 0.2  # 差分进化中的参数Cr
    # 求解
    # ones=np.ones((10,10))
    # pop=ones*(np.arange(1,11)[:,None])
    res = ea.optimize(algorithm,
                      prophet=None,
                      verbose=False,
                      drawing=False,
                      outputMsg=False,
                      drawLog=False,
                      saveFlag=False,
                      dirName='result')
    print(res)