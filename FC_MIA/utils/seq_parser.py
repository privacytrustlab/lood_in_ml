import numpy as np

h_message = '''
>>> You can use "min" or "max" to control the minimum and maximum values.
>>> constant
y = c
>>> linear
y = start_v + x * slope
>>> exp / exponential
y = start_v * power ** (x / interval)
>>> jump
y = start_v * power ** (max(idx - min_jump_pt, 0) // jump_freq)
'''


def continuous_seq(*args, **kwargs):
    '''
    >>> return a float to float mapping
    '''
    name = kwargs['name']
    max_v = kwargs['max'] if 'max' in kwargs else np.inf
    min_v = kwargs['min'] if 'min' in kwargs else -np.inf

    if name.lower() in ['h', 'help']:
        print(h_message)
        exit(0)
    elif name.lower() in ['constant',]:
        start_v = float(kwargs['start_v'])
        return lambda x: np.clip(start_v, a_min = min_v, a_max = max_v)
    elif name.lower() in ['linear',]:
        start_v = float(kwargs['start_v'])
        slope = float(kwargs['slope'])
        return lambda x: np.clip(start_v + x * slope, a_min = min_v, a_max = max_v)
    elif name.lower() in ['exp, exponential',]:
        start_v = float(kwargs['start_v'])
        power = float(kwargs['power'])
        interval = int(kwargs['interval']) if 'interval' in kwargs else 1
        return lambda x: np.clip(start_v * power ** (x / float(interval)), a_min = min_v, a_max = max_v)
    elif name.lower() in ['jump',]:
        start_v = float(kwargs['start_v'])
        power = float(kwargs['power'])
        min_jump_pt = int(kwargs['min_jump_pt'])
        jump_freq = int(kwargs['jump_freq'])
        return lambda x: np.clip(start_v * power ** (max(x - min_jump_pt + jump_freq, 0) // jump_freq), a_min = min_v, a_max = max_v)
    else:
        raise ValueError('Unrecognized name: %s'%name)

def discrete_seq(*args, **kwargs):
    '''
    >>> return a list of values
    '''
    name = kwargs['name']
    func = continuous_seq(*args, **kwargs)

    pt_num = int(kwargs['pt_num'])
    return [func(idx) for idx in range(pt_num)]

