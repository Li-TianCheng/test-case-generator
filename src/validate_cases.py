from pathlib import Path

import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr


def validate_positive_cases(output, constraints, symbol):
    equal_con = []
    not_equal_con = []
    gaussian_con = []  # [变量名字， mu， sigma]
    gaussian_output = []
    pop_constraints = []
    for i in range(len(constraints)):  # sympy不支持带[]和.的字符串解析
        constraints[i] = constraints[i].replace("[", "_")
        constraints[i] = constraints[i].replace("].", "_")
        constraints[i] = constraints[i].replace(".", "_")
        constraints[i] = constraints[i].replace("]", "")
        if constraints[i].find("==") != -1:
            equal_con.append(constraints[i].split("=="))
            pop_constraints.append(i)
        elif constraints[i].find("!=") != -1:
            not_equal_con.append(constraints[i].split("!="))
            pop_constraints.append(i)
        elif constraints[i].find("GAUSSIAN") != -1:
            con = constraints[i]
            pop_constraints.append(i)
            con_segment = con.split(",")
            var_name = con_segment[0].split("(")[1]
            mu = float(con_segment[1])
            sigma = float(con_segment[2].split(")")[0])
            gaussian_con.append([var_name, mu, sigma])
            gaussian_output.append([])
    for i in range(len(pop_constraints)):
        del constraints[pop_constraints.pop()]

    for i in range(len(symbol)):
        symbol[i] = symbol[i].replace("[", "_")
        symbol[i] = symbol[i].replace("].", "_")
        symbol[i] = symbol[i].replace(".", "_")
        symbol[i] = symbol[i].replace("]", "")

    var_dict = {}
    ok_flag = 1
    for case_index in range(len(output)):
        if len(output[case_index]) is not len(symbol):
            raise ValueError("第" + str(case_index) + "条样例的输入数据数量与变量数量不同，请检查")
        for i in range(len(symbol)):
            var_dict[symbol[i]] = output[case_index][i]
        for i in range(len(constraints)):
            expr = parse_expr(constraints[i], var_dict)
            if not (isinstance(expr, bool) or expr.is_Boolean):  # 我也不知道 为什么会有两种格式hhhh不懂了
                raise ValueError("第" + str(i) + "条不等式约束中有不在符号表中的变量")
            if not expr:
                print("第" + str(case_index) + "条样例的不符合第" + str(i) + "条不等式约束")
                ok_flag = 0
        for i in range(len(equal_con)):
            expr = parse_expr(equal_con[i][0], var_dict) - parse_expr(equal_con[i][1], var_dict)
            if not (isinstance(expr, float) or isinstance(expr, int) or expr.is_Number):
                raise ValueError("第" + str(i) + "条等式约束中有不在符号表中的变量")
            if abs(expr) > 0.001:
                print("第" + str(case_index) + "条样例的不符合第" + str(i) + "条等式约束")
                ok_flag = 0
        for i in range(len(not_equal_con)):
            expr = parse_expr(not_equal_con[i][0], var_dict) - parse_expr(not_equal_con[i][1], var_dict)
            if not (isinstance(expr, float) or isinstance(expr, int) or expr.is_Number):
                raise ValueError("第" + str(i) + "条不等于约束中有不在符号表中的变量")
            if abs(expr) < 0.001:
                print("第" + str(case_index) + "条样例的不符合第" + str(i) + "条不等于约束")
                ok_flag = 0
        for i in range(len(gaussian_con)):
            gaussian_output[i].append(var_dict[gaussian_con[i][0]])
    if ok_flag:
        print("果然都没啥问题")
    for i in range(len(gaussian_con)):
        plt.hist(gaussian_output[i], bins=50, color='steelblue')
        plt.show()


def validate_negative_cases(output, constraints, symbol):
    equal_con = []
    not_equal_con = []
    pop_constraints = []
    for i in range(len(constraints)):  # sympy不支持带[]和.的字符串解析
        constraints[i] = constraints[i].replace("[", "_")
        constraints[i] = constraints[i].replace("].", "_")
        constraints[i] = constraints[i].replace(".", "_")
        constraints[i] = constraints[i].replace("]", "")
        if constraints[i].find("==") != -1:
            equal_con.append(constraints[i].split("=="))
            pop_constraints.append(i)
        elif constraints[i].find("!=") != -1:
            not_equal_con.append(constraints[i].split("!="))
            pop_constraints.append(i)
        elif constraints[i].find("GAUSSIAN") != -1:
            pop_constraints.append(i)
    for i in range(len(pop_constraints)):
        del constraints[pop_constraints.pop()]

    for i in range(len(symbol)):
        symbol[i] = symbol[i].replace("[", "_")
        symbol[i] = symbol[i].replace("].", "_")
        symbol[i] = symbol[i].replace(".", "_")
        symbol[i] = symbol[i].replace("]", "")

    var_dict = {}
    ok_flag = 1
    for case_index in range(len(output)):
        case_pass_flag = 1
        if len(output[case_index]) is not len(symbol):
            raise ValueError("第" + str(case_index) + "条样例的输入数据数量与变量数量不同，请检查")
        for i in range(len(symbol)):
            var_dict[symbol[i]] = output[case_index][i]
        for i in range(len(constraints)):
            expr = parse_expr(constraints[i], var_dict)
            if not (isinstance(expr, bool) or expr.is_Boolean):
                raise ValueError("第" + str(i) + "条不等式约束中有不在符号表中的变量")
            if not expr:
                case_pass_flag = 0
        for i in range(len(equal_con)):
            expr = parse_expr(equal_con[i][0], var_dict) - parse_expr(equal_con[i][1], var_dict)
            if not (isinstance(expr, float) or isinstance(expr, int) or expr.is_Number):
                raise ValueError("第" + str(i) + "条等式约束中有不在符号表中的变量")
            if abs(expr) > 0.001:
                case_pass_flag = 0
        for i in range(len(not_equal_con)):
            expr = parse_expr(not_equal_con[i][0], var_dict) - parse_expr(not_equal_con[i][1], var_dict)
            if not (isinstance(expr, float) or isinstance(expr, int) or expr.is_Number):
                raise ValueError("第" + str(i) + "条不等于约束中有不在符号表中的变量")
            if abs(expr) < 0.001:
                case_pass_flag = 0
        if case_pass_flag:
            print("第" + str(case_index) + "条样例符合所有约束")
            ok_flag = 0
    if ok_flag:
        print("果然都有点问题")



if __name__ == "__main__":
    print('初始化中...')
    constraints_IO = open("./case/constraints.txt")
    constraints = constraints_IO.read().split("\n")
    constraints.pop()
    file0_IO = open("./case/positive/case0.txt")
    file0 = file0_IO.read().split("\n")
    symbol = []
    for con in file0:
        symbol.append(con.split("=")[0])
    symbol.pop()  # 最后一个是个回车
    # symbol = ["a", "b[0]", "b[1]", "b[2]", "s[0].a", "s[0].b", "s[0].c", "s[0].d[0]", "s[0].d[6]", "s[1].d[6]",
    #           "s[1].b"]

    print('正在验证正测试用例...')
    positive_output = []
    c_files = Path("./case/positive").glob('case*.txt')
    for file in c_files:
        positive_output.append([])
        f = open('./case/positive/' + file.name, 'r')
        item_list = f.read().split("\n")
        for item in item_list:
            if item is "":
                continue
            positive_output[-1].append(float(item.split("=")[1]))
    validate_positive_cases(positive_output, constraints.copy(), symbol)

    print('正在验证负测试用例...')
    negative_output = []
    c_files = Path("./case/negative").glob('case*.txt')
    for file in c_files:
        negative_output.append([])
        f = open('./case/negative/' + file.name, 'r')
        item_list = f.read().split("\n")
        for item in item_list:
            if item is "":
                continue
            negative_output[-1].append(float(item.split("=")[1]))
    validate_negative_cases(negative_output, constraints.copy(), symbol)
