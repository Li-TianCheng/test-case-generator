import argparse
import shutil

from find_solution import *
from gen_matrix import *
from gen_test_case import *


def print_error(value):
    print('error:', value)


def convert(output, symbol):
    contents = set()
    for i in range(len(output)):
        re = ""
        for j in range(len(output[i])):
            re = re + symbol[j] + '=' + str(output[i][j]) + '\n'
        contents.add(re)
    return contents


def task(bubble_matrix, constraint, bubble_input, idx, number=1):
    output = get_test_cases(bubble_matrix[idx].astype(np.int).copy(), constraint.copy(), bubble_input[idx].copy(),
                            number)
    c = convert(output, bubble_input[idx].copy())
    return c

def mutate(constraint, or_cons):
    while True:
        idx = random.randint(0, len(constraint)-1)
        if (constraint[idx] in or_cons):
            continue
        if constraint[idx].find('>') != -1:
            constraint[idx] = constraint[idx].replace('>', '<')
            return constraint
        if constraint[idx].find('<') != -1:
            constraint[idx] = constraint[idx].replace('<', '>')
            return constraint
        if constraint[idx].find('>=') != -1:
            constraint[idx] = constraint[idx].replace('>=', '<=')
            return constraint
        if constraint[idx].find('<=') != -1:
            constraint[idx] = constraint[idx].replace('<=', '>=')
            return constraint
        if constraint[idx].find('==') != -1:
            constraint[idx] = constraint[idx].replace('==', '!=')
            return constraint
        if constraint[idx].find('!=') != -1:
            constraint[idx] = constraint[idx].replace('!=', '==')
            return constraint

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test Case Generator')
    parser.add_argument('--name', help='c program name')
    parser.add_argument('--n', type=int, default=10000, help='test case number (default: 10000)')
    parser.add_argument('--r', type=float, default=0.5, help='ratio of positive case (default: 0.5)')
    parser.add_argument('--c', type=int, default=-1, help='thread number (default: number of cores(-1))')
    arg = parser.parse_args()
    kwargs = vars(arg)

    time_start = time.time()

    # parameters
    input_path = './resources/' + kwargs['name']
    NUM = kwargs['n']
    ratio = kwargs['r']
    pos_num = int(NUM * ratio)
    neg_num = NUM - pos_num
    if kwargs['c'] == -1:
        core_num = int(mp.cpu_count())
    else:
        core_num = kwargs['c']
    output_path = './case'

    # clear
    if os.path.exists('./case'):
        shutil.rmtree('./case')
    os.mkdir('./case')
    os.mkdir('./case/positive')
    os.mkdir('./case/negative')

    time_clear = time.time()
    print("清理完成（{:.3f}s）...".format(time_clear - time_start))

    # init
    input, constraint = analysis(input_path)
    input, pos_constraint, or_cons = revised(input, constraint)
    with open('./case/constraints.txt', 'w') as f:
        for line in pos_constraint:
            f.write(line + '\n')
    neg_constraint = mutate(pos_constraint.copy(), or_cons)
    pos_matrix = gen_matrix(input, pos_constraint)
    neg_matrix = gen_matrix(input, neg_constraint)
    pos_bubble_input, pos_bubble_matrix = matrix_split(pos_matrix, input, pos_constraint)
    neg_bubble_input, neg_bubble_matrix = matrix_split(neg_matrix, input, neg_constraint)
    pos_n = len(pos_bubble_matrix)
    neg_n = len(neg_bubble_matrix)
    pos_part_num = math.ceil(5*pos_num ** (1 / pos_n))
    neg_part_num = math.ceil(5*neg_num ** (1 / neg_n))
    pos_bubbles = list()
    pos_results = list()
    neg_bubbles = list()
    neg_results = list()
    pool = mp.Pool(core_num)

    time_init = time.time()
    print("文件解析及初始化完成（{:.3f}s）...".format(time_init - time_clear))

    # solute
    for i in range(len(pos_bubble_matrix)):
        pos_result = pool.apply_async(task, (pos_bubble_matrix, pos_constraint, pos_bubble_input, i, pos_part_num), error_callback=print_error)
        pos_results.append(pos_result)
    for i in range(len(neg_bubble_matrix)):
        neg_result = pool.apply_async(task, (neg_bubble_matrix, neg_constraint, neg_bubble_input, i, neg_part_num), error_callback=print_error)
        neg_results.append(neg_result)

    pool.close()
    pool.join()

    for result in pos_results:
        pos_bubbles.append(result.get())
    for result in neg_results:
        neg_bubbles.append(result.get())

    time_solute = time.time()
    print("约束求解完成（{:.3f}s）...".format(time_solute - time_init))

    # merge and generate
    pool = mp.Pool(core_num)
    pos_count = generator(pos_bubbles, output_path+'/positive', pos_num, pool)
    neg_count = generator(neg_bubbles, output_path+'/negative', neg_num, pool)
    pool.close()
    pool.join()

    count = pos_count + neg_count
    time_gen = time.time()
    print("生成测试用例完成（{:.3f}s）...".format(time_gen - time_solute))
    print(
        "总计用时：{:.3f}s\n测试用例个数：{}，正测试用例:{}个，负测试用例:{}个，正测试用例比例:{:.3f}%\n平均速度:{:.3f}个/s".format(time_gen-time_clear, count, pos_count, neg_count, pos_count/count*100, count/(time_gen-time_clear)))
