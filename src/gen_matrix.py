from pycparser import parse_file
import random
import numpy as np

 
def analysis_constraint(arg, constraint):
    if arg == None:
        return
    arg_left, arg_right, arg_op = analysis_constraint_helper(arg)
    analysis_constraint(arg_left, constraint)
    constraint.append(arg_op)
    constraint.append(' ')
    analysis_constraint(arg_right, constraint)

def analysis_constraint_helper(arg):
    if arg.__class__.__name__ == 'BinaryOp':
        return arg.left, arg.right, arg.op
    if arg.__class__.__name__ == 'FuncCall':
        n = arg.name.name
        n = n + '('
        for a in arg.args.exprs:
            _, _, op = analysis_constraint_helper(a)
            n = n + op + ','
        n = n[:-1]
        n = n + ')'
        return None, None, n
    if arg.__class__.__name__ == 'ArrayRef':
        _, _, n = analysis_constraint_helper(arg.name)
        _, _, f = analysis_constraint_helper(arg.subscript)
        return None, None, n + '[' + f + ']'
    if arg.__class__.__name__ == 'StructRef':
        f = arg.field.name
        _, _, n = analysis_constraint_helper(arg.name)
        return None, None, n + '.' + f
    if arg.__class__.__name__ == 'ID':
        return None, None, arg.name
    if arg.__class__.__name__ == 'Constant':
        return None, None, arg.value
    if arg.__class__.__name__ == 'UnaryOp':
        _, _, n = analysis_constraint_helper(arg.expr)
        return None, None, '-' + n

def analysis_Struct(arg, struct):
    body = dict()
    for arg in arg.type:
        l = len(arg.name)
        n = [arg.name]
        t = list()
        arg = arg.type
        _n, _t = analysis_Decl(arg, struct, n, t, l)
        for i in range(len(_n)):
            body[_n[i]] = _t[i]
    return body

def analysis_Decl(arg, struct, n, t, l):
    if arg.__class__.__name__ == 'IdentifierType':
        t.append(arg.names[0])
        return n, t
    if arg.__class__.__name__ == 'TypeDecl':
        _n = list()
        for i in n:
            _n.append(i+arg.declname[l:])
        n = _n
        n, t = analysis_Decl(arg.type, struct, n, t, l)
        return n, t
    if arg.__class__.__name__ == 'FuncDecl':
        return None, None
    if arg.__class__.__name__ == 'Struct':
        struct_name = arg.name
        _n = list()
        for i in n:
            for j in range(len(struct[struct_name])):
                _n.append(i+'.'+list(struct[struct_name].keys())[j])
                t.append(list(struct[struct_name].values())[j])
        n = _n
        return n, t
    if arg.__class__.__name__ == 'ArrayDecl':
        num = arg.dim.value
        arg = arg.type
        _n, _t = analysis_Decl(arg, struct, n, t, l)
        n_temp = list()
        for i in range(int(num)):
            for j in n:
                curr = j+'['+str(i)+']'
                for k in range(len(_n)):
                    n_temp.append(curr+_n[k][l:])
                    t.append(_t[k])
        n = n_temp
        return n, t
    if arg.__class__.__name__ == 'PtrDecl':
        arg = arg.type
        _n, _t = analysis_Decl(arg, struct, n, t, l)
        t_temp = list()
        for k in _t:
            t_temp.append(k+'_ptr')
        n = _n
        t = t_temp
        return n, t

def analysis(path):
    ast = parse_file(path)
    input = dict()
    struct = dict()
    constraint = list()
    for arg in ast:
        if arg.__class__.__name__ == 'Typedef':
            body = analysis_Struct(arg.type, struct)
            struct[arg.name] = body
        if arg.__class__.__name__ == 'Decl':
            l = len(arg.name)
            n = [arg.name]
            t = list()
            arg = arg.type
            _n, _t = analysis_Decl(arg, struct, n, t, l)
            if _n != None:
                for i in range(len(_n)):
                    input[_n[i]] = _t[i]
        if arg.__class__.__name__ == 'FuncDef':
            arg = arg.body
            for c in arg.block_items:
                con = list()
                analysis_constraint(c, con)
                con = " "+"".join(con)
                constraint.append(con)
    return input, constraint

def revised(input, constraint):
    _del = set()
    or_cons = list()
    for c in constraint:
        if '_LENGTH' in c:
            _del.add(c)
            upper = 999
            lower = 0
            field = None
            while '_LENGTH' in c:
                idx = c.find('_LENGTH')
                c = c[idx+8:]
                idx = c.find(')')
                field = c[:idx]
                c = c[idx+2:]
                idx = c.find('&&')
                if c[0] == '<':
                    if c[1] == ' ':
                        upper = int(c[1: idx]) - 1
                    if c[1] == '=':
                        upper = int(c[3: idx])
                if c[0] == '>':
                    if c[1] == ' ':
                        lower = int(c[1: idx]) + 1
                    if c[1] == '=':
                        lower = int(c[3: idx])
                c = c[idx+3:]
            num = random.randint(lower, upper)
            t = input.pop(field)
            idx = t.find('_ptr')
            t = t[: idx]
            for i in range(num):
                input[field+'['+str(i)+']'] = t
        else:
            if '||' in c:
                _del.add(c)
                _c = ''.join(c)
                temp = list()
                while '||' in _c:
                    idx = _c.find('||')
                    temp.append(_c[:idx])
                    _c = _c[idx+2:]
                temp.append(_c)
                index = random.randint(0, len(temp)-1)
                constraint.append(temp[index])
                or_cons.append(temp[index])
            if '&&' in c and c not in _del:
                _del.add(c)
                _c = ''.join(c)
                while '&&' in _c:
                    idx = _c.find('&&')
                    constraint.append(_c[:idx])
                    if c in or_cons:
                        or_cons.append(_c[:idx])
                    _c = _c[idx+2:]
                constraint.append(_c)
                if c in or_cons:
                    or_cons.remove(c)
                    or_cons.append(_c)
    for c in _del:
        constraint.remove(c)
    return input, constraint, or_cons

def gen_matrix(input, constraint):
    matrix = np.zeros((len(constraint)+1, len(input)))
    for i in range(len(constraint)):
        j = 0
        n = ''
        if 'GAUSSIAN' in constraint[i]:
            idx = constraint[i].find(',')
            n = constraint[i][10: idx]
        for var in input:
            if ' '+var+' ' in constraint[i]:
                matrix[i, j] = 1
            if 'GAUSSIAN' in constraint[i] and var == n:
                matrix[i, j] = 1
            j = j + 1
    j = 0
    for var in input:
        if input[var] == 'double':
            matrix[-1, j] = 1
        j = j + 1
    return matrix

def find(root, num):
    if root[int(num)] != num:
        root[int(num)] = find(root, int(root[num]))
    return root[int(num)]

def merge(root, num1, num2):
    link(root, find(root, num1), find(root, num2))

def link(root, num1, num2):
    if num1 < num2:
        root[int(num2)] = num1
    else:
        root[int(num1)] = num2

def matrix_split(matrix, input, constraint):
    root = np.zeros(len(input))
    for i in range(len(input)):
        root[i] = i
    for i in range(matrix.shape[0]-1):
        for j in range(matrix.shape[1]):
            for k in range(j+1, matrix.shape[1]):
                if matrix[i, j] + matrix[i, k] == 2:
                    merge(root, j, k)
    bubble_input = dict()
    bubble_matrix = dict()
    for i in range(root.shape[0]):
        if bubble_input.get(find(root, i)) is None:
            temp_matrix = list()
            temp_input = dict()
            temp_matrix.append(matrix[:,i])
            n = list(input.keys())[i]
            temp_input[n] = input[n]
            bubble_input[root[i]] = temp_input
            bubble_matrix[root[i]] = temp_matrix
        else:
            bubble_matrix[int(find(root, i))].append(matrix[:,i])
            n = list(input.keys())[i]
            bubble_input[int(find(root, i))][n] = input[n]
    for i in bubble_matrix:
        bubble_matrix[i].append(np.zeros(len(bubble_matrix[i][0])))
        bubble_matrix[i] = np.array(bubble_matrix[i]).T
    bubble_matrix = list(bubble_matrix.values())
    bubble_input = list(bubble_input.values())
    _bubble_input = list()
    for b in bubble_input:
        _bubble_input.append(list(b.keys()))
    return _bubble_input, bubble_matrix

if __name__ == "__main__":
    input, constraint = analysis('./resources/t2.c')
    input, constraint, or_cons = revised(input, constraint)
    matrix = gen_matrix(input, constraint)
    bubble_input, bubble_matrix = matrix_split(matrix, input, constraint)
    print(constraint)
