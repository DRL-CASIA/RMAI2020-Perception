

def parse_model_config(path):  # 读取cfg文件  # 每个block中分为按各个type区分，分别以字典形式记录cfg文件中各层的配置值
    """Parses the yolo-v3 layer configuration file and returns module definitions"""
    file = open(path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    module_defs = []
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            module_defs.append({})
            module_defs[-1]['type'] = line[1:-1].rstrip()  # 倒数第一个(即新增的)module def的type key值
            if module_defs[-1]['type'] == 'convolutional':
                module_defs[-1]['batch_normalize'] = 0  # 先设置为0
        else:
            key, value = line.split("=")
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    return module_defs  # 字典组成的数组  [{type:...,key:...,key:...,key:...},{type:...,key:...,key:...,key:...},{type:...,key:...,key:...,key:...},...]

def parse_data_config(path):  # 读取coco.data时使用?
    """Parses the data configuration file"""
    options = dict()
    # options['gpus'] = '0,1,2,3'
    # options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


