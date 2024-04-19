def required_kernel(in_size: int, out_size:int, stride=1, padding=1):
    assert in_size > 0, "Input size must be greater than 0"
    assert out_size > 0, "Output size must be greater than 0"
    assert in_size >= out_size, "Input size must be greater than or equal to output size"
    assert stride > 0, "Stride must be greater than 0"
    assert padding >= 0, "Padding must be greater than or equal to 0"
    
    return (1-out_size)*stride+in_size+2*padding

def get_output(in_size: int, kernel_size: int, stride=1, padding=1):
    assert in_size > 0, "Input size must be greater than 0"
    assert kernel_size > 0, "Kernel size must be greater than 0"
    assert stride > 0, "Stride must be greater than 0"
    assert padding >= 0, "Padding must be greater than or equal to 0"
    
    return (in_size-kernel_size+2*padding)//stride+1

print(get_output(224, 3, 1, 1))