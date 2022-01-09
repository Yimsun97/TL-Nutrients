# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import geatpy as ea


# 导入自定义问题接口
class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, func, Dim, lb, ub, n_group):

        # 存储func
        self.func = func
        self.Dim = Dim
        self.n_group = n_group

        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 1  # 初始化M（目标维数）
        maxormins = [1]  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        # Dim_nh = 1 # 初始化Dim（决策变量维数）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        # lb_nh = [-1]  # 决策变量下界
        # ub_nh = [2]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def aimFunc(self, pop):  # 目标函数
        Vars = pop.Phen  # 得到决策变量矩阵
        x = np.zeros((self.n_group, 1))

        i = 0
        for value in Vars:
            x[i] = self.func(value)
            i += 1

        pop.ObjV = x  # 计算目标函数值，赋值给pop种群对象的ObjV属性


def parameter_define(func, Dim, lb, ub, n_group, n_iter, pco, pm, seed=1024):
    """ 定义优化问题

    """

    # 随机数种子设置
    np.random.seed(seed=seed)

    """===============================实例化问题对象==========================="""
    problem = MyProblem(func, Dim, lb, ub, n_group)  # 生成问题对象
    """=================================种群设置=============================="""
    Encoding = 'BG'  # 编码方式
    NIND = n_group  # 种群规模
    Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
    population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
    """===============================算法参数设置============================="""
    myAlgorithm = ea.soea_SEGA_templet(problem, population)  # 实例化一个算法模板对象
    myAlgorithm.MAXGEN = n_iter  # 最大进化代数
    myAlgorithm.recOper.XOVR = pco  # 设置交叉概率
    myAlgorithm.mutOper.Pm = pm  # 设置变异概率
    myAlgorithm.logTras = 1  # 设置每隔多少代记录日志，若设置成0则表示不记录日志
    myAlgorithm.verbose = True  # 设置是否打印输出日志信息
    myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
    """==========================调用算法模板进行种群进化========================"""
    [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
    # 保留前5个最优的解
    # top_five_index = np.argsort(population.ObjV * problem.maxormins, axis=0)[:5]
    # top_five_pop = population[top_five_index]
    BestIndi.save()  # 把最优个体的信息保存到文件中
    """=================================输出结果=============================="""
    print('评价次数：%s' % myAlgorithm.evalsNum)
    print('时间已过 %s 秒' % myAlgorithm.passTime)
    if BestIndi.sizes != 0:
        print('最优的目标函数值为：%s' % BestIndi.ObjV[0][0])
        print('最优的控制变量值为：')
        for i in range(BestIndi.Phen.shape[1]):
            print(BestIndi.Phen[0, i])
    else:
        print('没找到可行解。')

    return myAlgorithm


if __name__ == '__main__':
    # 设置随机种子
    np.random.seed(1024)

    f = lambda x: x[0] * np.sin(10 * np.pi * x[1]) + 2.0
    results, _ = parameter_define(f, 2, [-1, -1], [2, 2], 40, 25, 0.7, 0.01, 1024)
