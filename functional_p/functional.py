import sys


def sequential_map(*args):
    *funcs, container = args
    for func in funcs:
        container = map(func, container)
        #print(container)
    return list(container)
    

def consensus_filter(*args):
    *funcs, container = args
    for func in funcs:
        container = filter(func, container)
    return list(container)


def conditional_reduce(conditional_func, reduce_func, container):
    filtered_container = list(filter(conditional_func, container))
    rest = filtered_container[0]
    for i in range(1, len(filtered_container)):
        rest = reduce_func(rest, filtered_container[i])
    return rest


def func_chain(*functions):
    def _pipeline(arg):
        result = arg
        for func in functions:
            result = func(result)
        return result
    return _pipeline


def sequential_map_upd(*args):
    *funcs, container = args
    new_function = func_chain(*funcs)
    return list(map(new_function, container))


def my_print(*args, sep=' ', end='\n', file=sys.stdout):
    items = map(str, args)
    string = sep.join(items) + end
    file.write(string)


def multiple_partial(*args, **kwargs):
    answer = []
    for func in args:
        def new_func(x, func=func):
            result = func(x, **kwargs)
            return result
        new_func.__name__ = func.__name__
        answer.append(new_func)
    return answer
