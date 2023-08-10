import configparser
import os

def get_fit_param(filename = 'config.ini'):
    # создание объекта конфигурации
    config = configparser.ConfigParser()
    # чтение конфигурации из файла
    
    if not os.path.exists(filename):
        raise ValueError("Configuration file doesn't exist.")
    
    config.read(filename)
    if not config.has_section('PARAMETERS'):
        raise ValueError("'PARAMETERS' section not found in the configuration.")
    
    params = {}

    # получение значения параметров
    params['model_path'] = os.path.normpath(config['PARAMETERS']['model_path'])
    params['dataset_path'] = os.path.normpath(config['PARAMETERS']['dataset_path'])
    params['epochs'] = int(config['PARAMETERS']['epochs'])
    params['batch_size'] = int(config['PARAMETERS']['batch_size'])
    params['split_size'] = float(config['PARAMETERS']['split_size'])

    input_shape = config['PARAMETERS']['input_shape']
    params['input_shape'] = tuple(map(int, input_shape.split(',')))

    output_shape = config['PARAMETERS']['output_shape']
    params['output_shape'] = tuple(map(int, output_shape.split(',')))

    res_cols = config['PARAMETERS']['res_cols']
    params['res_cols'] = list(map(int,res_cols[1:-1].split(','))) if res_cols != '[]' else []

    params['loss']= config['PARAMETERS']['loss']
    params['metric']= config['PARAMETERS']['metric']
    params['drop_na'] = config['PARAMETERS']['drop_na'] == 'True'
    params['col_names'] = config['PARAMETERS']['col_names'] == 'True'
    #res_cols = tuple(map(int, res_cols.split(',')))
    #learning_rate = float(config['PARAMETERS']['learning_rate'])
    return params

def save_fit_param(fn_conf,model_path, dataset_path, epochs, 
                   batch_size, split_size, input_shape, output_shape, 
                   res_cols, loss_func,metric, drop_na, col_names):
    # создание объекта конфигурации
    config = configparser.ConfigParser()
 
    # добавление значений в секцию
    config['PARAMETERS'] = {
        'model_path': model_path,
        'dataset_path': dataset_path,
        'epochs': epochs,
        'batch_size': batch_size,
        'split_size': split_size,
        'input_shape': input_shape,
        'output_shape': output_shape,
        'res_cols': res_cols,
        'drop_na': drop_na,
        'col_names': col_names,
        'loss' : loss_func,
        'metric': metric
    }
    # запись конфигурации в файл
    with open(fn_conf, 'w') as configfile:
        config.write(configfile)
