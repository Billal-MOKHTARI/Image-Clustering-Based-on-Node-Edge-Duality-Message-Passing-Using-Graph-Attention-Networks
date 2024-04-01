def required_kernel(in_size: int, out_size:int, stride=1, padding=1):
    assert in_size > 0, "Input size must be greater than 0"
    assert out_size > 0, "Output size must be greater than 0"
    assert in_size >= out_size, "Input size must be greater than or equal to output size"
    assert stride > 0, "Stride must be greater than 0"
    assert padding >= 0, "Padding must be greater than or equal to 0"
    
    return (1-out_size)*stride+in_size+2*padding

def convert_to_int(input_list):
    result = []
    for item in input_list:
        if isinstance(item, list):
            result.append(convert_to_int(item))
        elif item.isdigit():  # Check if the string represents a number
            result.append(int(item))
        else:
            result.append(item)
    return result

def create_dictionary(keys, values):
    return dict(zip(map(tuple, keys), values))

def bi_operator(op, a, b):
    if op == '==':
        
        return a == b
    elif op == '!=':
        return a != b
    elif op == '>':
        return a > b
    elif op == '>=':
        return a >= b
    elif op == '<':
        return a < b
    elif op == '<=':
        return a <= b
    elif callable(op):
        op(a, b)
