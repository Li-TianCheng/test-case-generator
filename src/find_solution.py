import copy
import math
import random
import time

import numpy as np
import sympy
from sympy import solveset, S, Intersection
from sympy.parsing.sympy_parser import parse_expr

# con 大概都是constraints的意思。。

Status_Storage = []
Recursion_Mark = []


def get_feasible_interval_for_int(interval):
    # 输入为单个interval，返回两个数的list，是可以用randrange采样的左闭右开区间
    INT_MAX = 1e10
    if interval.args[0].is_infinite:
        left = -INT_MAX
    else:
        if interval.args[2] == False:  # 区间左侧为闭
            left = math.ceil(interval.args[0])
        else:
            left = math.floor(interval.args[0]) + 1  # 采样时需要输入左闭右开区间
    if interval.args[1].is_infinite:
        right = INT_MAX
    else:
        if interval.args[3] == False:
            right = math.floor(interval.args[1]) + 1
        else:
            right = math.ceil(interval.args[1])
    return [left, right]


def get_random_from_interval(interval, type, count=1):
    if interval.is_FiniteSet:
        index = random.randrange(len(interval.args))
        return interval.args[index]

    FLOAT_MAX = 1e10
    INT_MAX = int(1e10)
    if interval.boundary is S.EmptySet:  # 负无穷到正无穷，松弛变量
        if type == 0:
            return random.randrange(-INT_MAX, INT_MAX)
        elif type == 1:
            return random.uniform(-FLOAT_MAX, FLOAT_MAX)

    if type == 0:  # int 处理有带无穷的区间，就只在带无穷的区间取值
        if interval.is_Union:  # 如果是多区间并集且第一个或最后一个区间有无穷
            interval = interval.args  # Union的args是由interval组成的tuple
            if math.isinf(interval[0].args[0]):
                return random.randrange(-INT_MAX, math.ceil(interval[0].args[1]))
            elif math.isinf(interval[-1].args[1]):
                return random.randrange(math.ceil(interval[-1].args[0]), INT_MAX)
        else:  # 单个区间，不管有没有无穷，可以直接扔给get_feasible_interval_for_int处理
            [l, r] = get_feasible_interval_for_int(interval)
            if l >= r:  # 如果上一步是类似(5,6)这样没有可行解的区间，lr值就会一样
                return None
            return random.randrange(l, r)
    elif type == 1:  # double
        if interval.is_Union:
            interval = interval.args
            if math.isinf(interval[0].args[0]):
                return random.uniform(-FLOAT_MAX, interval[0].args[1])
            elif math.isinf(interval[-1].args[1]):
                return random.uniform(interval[-1].args[0], FLOAT_MAX)
        else:
            if math.isinf(interval.args[0]):
                return random.uniform(-FLOAT_MAX, interval.args[1])
            elif math.isinf(interval.args[1]):
                return random.uniform(interval.args[0], FLOAT_MAX)
            else:
                return random.uniform(interval.args[0], interval.args[1])
    # 以上是区间里有无穷的特殊情况，和只有单个区间
    # 以下是多个区间组成，均为实数边界
    boundary_list = []
    if type == 0:  # int
        for single_inter in interval:
            boundary_list.append(get_feasible_interval_for_int(single_inter))
        boundary_array = np.asarray(boundary_list)
        pdf = np.cumsum(boundary_array[:, 1] - boundary_array[:, 0])
        random_index = random.randrange(pdf[-1]) + 1
        interval_index = np.where(pdf >= random_index)[0][0]
        offset = pdf[interval_index] - random_index + 1
        return boundary_list[interval_index][1] - offset
    elif type == 1:
        for single_inter in interval:
            boundary_list.append([float(single_inter.args[0]), float(single_inter.args[1])])
        boundary_array = np.asarray(boundary_list)
        pdf = np.cumsum(boundary_array[:, 1] - boundary_array[:, 0])
        random_index = random.uniform(0, pdf[-1])
        interval_index = np.where(pdf >= random_index)[0][0]
        offset = pdf[interval_index] - random_index
        return boundary_list[interval_index][1] - offset


def check_not_equal_constraints(not_equal_con, var_dict):
    # 有!=的==则返回False
    for con in not_equal_con:
        func = sympy.Eq(parse_expr(con[0], var_dict), parse_expr(con[1], var_dict))
        if func:  # 给sympy任何一个式子他都会先进行简化，比如上面这个全是常量的等式，就会被简化成True或者False
            return False
    return True


def save_status(A, var_dict, var_interval, gaussian_con):
    global Status_Storage
    global Recursion_Mark
    Status_Storage.append(copy.deepcopy([A, var_dict, var_interval, gaussian_con]))
    Recursion_Mark.append(0)


def clear_status():
    global Status_Storage
    global Recursion_Mark
    Status_Storage = []
    Recursion_Mark = []


def recursion(last_status_invalid=0, max_recursion=3):  # 在同一点最多回溯次数，超过就多回溯一级
    global Status_Storage
    global Recursion_Mark
    if last_status_invalid:
        Status_Storage.pop()
        Recursion_Mark.pop()
    Recursion_Mark[-1] += 1
    if Recursion_Mark[-1] > max_recursion:
        available_recur = [i for i in range(len(Recursion_Mark)) if Recursion_Mark[i] < max_recursion]
        if len(available_recur) is 0:
            raise ValueError("好像无解，要不再试一下")
        recur_index = available_recur[-1]
        Recursion_Mark = Recursion_Mark[:recur_index + 1]
        Status_Storage = Status_Storage[:recur_index + 1]
        Recursion_Mark[recur_index] += 1
        return copy.deepcopy(Status_Storage[recur_index])
    else:
        return copy.deepcopy(Status_Storage[-1])


def chain_reaction(A, var_no, var_interval, constraints, symbol):
    if not (A.shape[1] > 1 and A.shape[0] > 2):
        return [A, var_interval]
    new_var_interval = copy.deepcopy(var_interval)
    var_index = np.where(A[-1, :-1] == var_no)[0][0]
    if var_interval[var_no].is_Union:
        lbound = var_interval[var_no].args[0].args[0]
        rbound = var_interval[var_no].args[-1].args[1]
    else:
        lbound = var_interval[var_no].args[0]
        rbound = var_interval[var_no].args[1]

    row_sum = np.sum(A[:-2, :-1], 1)
    con_index = np.where((row_sum == 2) * A[:-2, var_index] == 1)[0]  # 是包含var_no和另一个变量的约束
    con_no = A[con_index, -1]
    A[:-2, var_index] = np.zeros(A[:-2, var_index].shape)
    for i in range(con_no.shape[0]):
        func = parse_expr(constraints[con_no[i]])
        func = sympy.solve(func, symbol[var_no])  # 可以把当前变量整理到式子左侧
        if ((not lbound.is_infinite) and (func.rel_op is '<' or func.rel_op is '<=')):  # 说明可以给式子右侧的一坨确定一个下界
            new_var_index = np.where(A[con_index[i], :-1] == 1)[0][0]
            result = solveset(func.subs({symbol[var_no]: lbound}), domain=sympy.S.Reals)
            new_var_interval[new_var_index] = Intersection(new_var_interval[new_var_index], result)
        elif ((not rbound.is_infinite) and (func.rel_op is '>' or func.rel_op is '>=')):
            new_var_index = np.where(A[con_index[i], :-1] == 1)[0][0]
            result = solveset(func.subs({symbol[var_no]: rbound}), domain=sympy.S.Reals)
            new_var_interval[new_var_index] = Intersection(new_var_interval[new_var_index], result)
    old_new_var_interval = new_var_interval
    for i in range(len(var_interval)):
        if old_new_var_interval[i] != var_interval[i]:
            [A, new_var_interval] = chain_reaction(A, i, new_var_interval, constraints, symbol)
    return [A, new_var_interval]


def get_single_case(A, type, constraints, gaussian_con, not_equal_con, symbol, max_recursion):
    clear_status()
    var_interval = [sympy.Interval(float("-inf"), float("inf"))] * len(symbol)
    var_dict = {}
    status_changed = 0
    while (A.shape[1] > 1 and A.shape[0] > 2):
        # 先解只有单变量的约束
        row_sum = np.sum(A[:-2, :-1], 1)
        con_index = np.where(row_sum == 1)[0]
        single_var_con = A[con_index, :]
        A = np.delete(A, con_index, axis=0)
        for single_con in single_var_con:
            var_no = A[-1, np.where(single_con == 1)[0][0]]
            single_con_str = constraints[single_con[-1]]
            if isinstance(single_con_str, list):  # 说明是等式，list里俩东西分别是等号左右
                func = sympy.Eq(parse_expr(single_con_str[0], var_dict), parse_expr(single_con_str[1], var_dict))
            else:
                func = parse_expr(single_con_str, var_dict)
            result = solveset(func, domain=sympy.S.Reals)
            total_result = Intersection(result, var_interval[var_no])
            if total_result is S.EmptySet:
                [A, var_dict, var_interval, gaussian_con] = recursion(max_recursion=max_recursion)
                status_changed = 0
                break
            else:
                var_interval[var_no] = total_result
                status_changed = 1
                var_interval = chain_reaction(copy.deepcopy(A), var_no, var_interval.copy(), constraints, symbol)[1]

        # 每个status的记录是以 区间计算完了，没有空集 为节点记录的，
        # 如果有空集说明刚才随机赋值的值不好，正好在这恢复状态重取一个
        if status_changed:
            save_status(A, var_dict, var_interval, gaussian_con)
            status_changed = 0

        # 采样时如果约束里有高斯，则优先对这种变量采样
        if len(gaussian_con) > 0:
            [var_no, mu, sigma] = gaussian_con.pop(0)
            var_range = var_interval[var_no]
            random_value = get_random_from_interval(var_range, type[var_no])
            if random_value is None:  # 先看一下高斯约束的变量有没有解，要是没有解那就不用玩了
                [A, var_dict, var_interval, gaussian_con] = recursion(max_recursion=max_recursion,
                                                                      last_status_invalid=1)
                continue
            failure_counter = 0
            if var_no in A[-1, :]:  # 如果没有的话说明这变量只有高斯约束，没有其他约束，在上面删无约束变量的时候被删掉了，就不要再删了
                var_index = np.where(A[-1, :] == var_no)[0][0]
                A = np.delete(A, var_index, axis=1)
            while 1:
                random_value = np.random.normal(float(mu), float(sigma))
                if type[var_no] == 0:
                    random_value = int(random_value + 0.5)  # 四舍五入
                if random_value in var_range:
                    var_interval[var_no] = random_value
                    var_dict[symbol[var_no]] = random_value
                    break
                failure_counter += 1
                if failure_counter > 10:
                    sigma = float(sigma) * 4
                    failure_counter = 5
            continue  # 就跳过下面采样的步骤了

        # 然后是挑约束最多的变量，随机赋个值
        column_sum = np.sum(A[:-2, :-1], 0)
        var_index = np.where(column_sum == np.max(column_sum, 0))[0][0]
        var_no = A[-1, var_index]  # 原始的变量序号
        A = np.delete(A, var_index, axis=1)
        var_range = var_interval[var_no]
        random_value = get_random_from_interval(var_range, type[var_no])
        if random_value is None:
            [A, var_dict, var_interval, gaussian_con] = recursion(max_recursion=max_recursion, last_status_invalid=1)
            continue  # 本来这能跳到if status_changed那最好，但是不行，从头也一样
        var_interval[var_no] = random_value
        var_dict[symbol[var_no]] = random_value


    # 如果到这高斯约束还有没解决的，说明A列数直接<=1，则bubble里只有这一种约束，约束着这一个变量
    if len(gaussian_con) > 0:
        for item in gaussian_con:
            [var_no, mu, sigma] = item
            random_value = np.random.normal(float(mu), float(sigma))
            if type[var_no] == 0:
                random_value = int(random_value + 0.5)  # 四舍五入
            var_interval[var_no] = random_value
            var_dict[symbol[var_no]] = random_value

    # 把没采样的变量都采个样
    output = []
    for i in range(len(var_interval)):
        if isinstance(var_interval[i], int) or isinstance(var_interval[i], float):
            output.append(var_interval[i])
        elif var_interval[i].is_Atom:  # 大概率是个Rational，如果约束里有乘法的话
            if type[i] is 0:  # int
                output.append(int(var_interval[i]))
            else:
                output.append(float(var_interval[i]))
        elif var_interval[i].is_Interval or var_interval[i].is_Union or var_interval[i].is_FiniteSet:
            random_value = get_random_from_interval(var_interval[i], type[i])
            output.append(random_value)
            var_dict[symbol[i]] = random_value
        else:
            raise TypeError("你看看刚才这是个啥")
    if None in output:
        return None
    if check_not_equal_constraints(not_equal_con, var_dict):  # 返回false说明样例不合格，直接不append就行了
        return output
    else:
        return None


def get_test_cases(A, constraints, symbol, number=1):
    # 输入的A:  系数  或的组号
    #          变量类型
    A = np.asarray(A)
    var_type = A[-1, :]
    b = np.arange(0, A.shape[0]).reshape((-1, 1))  # 约束序号，因为计算过程会删行，方便在字符串里找对应约束
    c = np.arange(0, A.shape[1])  # 变量序号
    A = np.hstack((A[:, :-1], b))
    A = np.vstack((A, c))
    # A的结构    系数    约束编号
    #           变量类型
    #           变量编号
    row_sum = np.sum(A[:-2, :-1], 1)  # 删掉和当前变量没关系的约束
    no_con_index = np.where(row_sum == 0)[0]
    A = np.delete(A, no_con_index, axis=0)

    not_equal_con = []  # 傻屌sympy, ==和!=需单独处理 -> !=最后再过滤
    gaussian_con = []  # [变量序号， mu， sigma]
    for i in range(len(constraints)):  # sympy不支持带[]和.的字符串解析
        constraints[i] = constraints[i].replace("[", "_")
        constraints[i] = constraints[i].replace("].", "_")
        constraints[i] = constraints[i].replace(".", "_")
        constraints[i] = constraints[i].replace("]", "")
    for i in A[:-2, -1]:  # 因为前面已经删了几行A，对被删掉的约束这里就不再处理了
        if constraints[i].find("==") != -1:
            constraints[i] = constraints[i].split("==")
        elif constraints[i].find("!=") != -1:
            not_equal_con.append(constraints[i].split("!="))
            con_no = np.where(A[:-2, -1] == i)[0][0]
            A = np.delete(A, con_no, axis=0)
        elif constraints[i].find("GAUSSIAN") != -1:
            con_no = np.where(A[:-2, -1] == i)[0][0]
            var_index = np.where(A[con_no, :] == 1)[0][0]
            var_no = A[-1, var_index]
            A = np.delete(A, con_no, axis=0)
            con_segment = constraints[i].split(",")
            mu = con_segment[1]
            sigma = con_segment[2].split(")")[0]
            gaussian_con.append([var_no, mu, sigma])

    for i in range(len(symbol)):
        symbol[i] = symbol[i].replace("[", "_")
        symbol[i] = symbol[i].replace("].", "_")
        symbol[i] = symbol[i].replace(".", "_")
        symbol[i] = symbol[i].replace("]", "")

    # 去除无关变量
    column_sum = np.sum(A[:-2, :-1], 0)
    slack_pos = np.hstack((column_sum == 0, False))
    A = np.delete(A, A[-1, slack_pos], axis=1)

    final_output = []
    max_recursion = 50
    while len(final_output) < number:
        output = get_single_case(A, var_type, constraints.copy(), copy.deepcopy(gaussian_con), not_equal_con.copy(),
                                 symbol.copy(),
                                 max_recursion)
        if output != None:
            final_output.append(output)
            max_recursion = 999  # 反正碰碰运气，有一个结果的话总能出来更多的
    return final_output


if __name__ == "__main__":
    # A = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
    #      [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #      [1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0],
    #      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    #      [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]]
    # constraints = ["a > 5",
    #                "a < 10",
    #                "b[1] > b[2]",
    #                "b[0] < b[1]",
    #                "a + b[0] != s[0].a",
    #                "s[0].d[6] + s[1].d[6] == s[0].b * s[1].b",
    #                "GAUSSIAN(s[0].c, 1.0, 1.0);"]
    # symbol = ["a", "b[0]", "b[1]", "b[2]", "s[0].a", "s[0].b", "s[0].c", "s[0].d[0]", "s[0].d[6]", "s[1].d[6]",
    #           "s[1].b"]

    A = [[1, 1, 0, 0, 0, 0],
         [0, 1, 1, 0, 0, 0],
         [0, 0, 1, 1, 0, 0],
         [0, 0, 0, 1, 1, 0],
         [1, 0, 0, 0, 0, 0],
         [0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 0, 0]]
    constraints = ["a<b",
                   "b<c",
                   "c<d",
                   "d<e",
                   "a>=-10",
                   "e<=10"]
    symbol = ["a", "b", "c", "d", "e"]
    time_start = time.time()
    output = get_test_cases(A, constraints, symbol, 10)
    time_end = time.time()
    print('生成', len(output), '个测试样例,用时', time_end - time_start, 's')
