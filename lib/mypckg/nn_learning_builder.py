import tensorflow as tf
import torch
import torch.nn as nn
import sys
import os
import shutil
import importlib.util


def validate_keras_model(model, input_shape, output_shape):
    # Проверяем, является ли объект model экземпляром класса keras.models.Model
    if not isinstance(model, tf.keras.models.Model):
        print("The provided object is not a Keras model.")
        return False
    # Проверяем наличие слоев
    if len(model.layers) == 0:
        print("The model does not contain any layers.")
        return False
    # Проверяем корректность входных и выходных размерностей
    try:
        res = model.predict_on_batch(tf.keras.backend.zeros(input_shape))
    except Exception as e:
        print("Failed to make predictions on empty input data.")
        print("Details:", str(e))
        return False
    return res.shape == output_shape

def validate_torch_model(model, input_shape, output_shape):
    # Проверяем, является ли объект model экземпляром класса torch.nn.Module
    if not isinstance(model, nn.Module):
        print("The provided object is not a PyTorch model.")
        return False
    # Проверяем наличие слоев
    if len(list(model.modules())) <= 1:
        print("The model does not contain any submodules.")
        return False
    # Проверяем корректность входных и выходных размерностей
    try:
        input_tensor = torch.zeros(input_shape)
        with torch.no_grad():
            res = model(input_tensor)
    except Exception as e:
        print("Failed to make predictions on empty input data.")
        print("Details:", str(e))
        return False
    return tuple(res.shape) == output_shape

def validate_model(model, input_shape, output_shape):
    if isinstance(model, tf.keras.models.Model):
        if not validate_keras_model(model,input_shape, output_shape):
            raise ValueError("Model must agree with the shape parameters")
    elif isinstance(model, nn.Module):
        if not validate_torch_model(model,input_shape, output_shape):
            raise ValueError("Model must agree with the shape parameters")
    else:
        raise ValueError("Model must be PyTorch model or Keras model.")

def get_model_info(model, loss):
    loss_func = {
        'mean_squared_error': 'MSELoss',
        'mean_absolute_error':'L1Loss',
        'categorical_crossentropy': 'CrossEntropyLoss'
    }
    loss_fn = loss
    template_path = '_train_template.py'

    if isinstance(model, tf.keras.models.Model):
        template_path = 'tfkeras' + template_path
    elif isinstance(model, nn.Module):
        template_path = 'torchnn' + template_path
        loss_fn = loss_func[loss]

    return template_path, loss_fn

def validate_params(model_file, dst_dir, data_file, epochs,
                             batch_size, split_size, input_shape, output_shape, 
                             res_cols, drop_na, col_names):
    if not os.path.isfile(model_file):
        raise ValueError("Invalid model file path.")
    if not os.path.isdir(dst_dir):
        raise ValueError("Invalid destination directory path.")
    if not os.path.isfile(data_file):
        raise ValueError("Invalid data file path.")
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("Epochs must be a positive integer.")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if not isinstance(split_size, float) or split_size >= 100 or split_size <= 0:
        raise ValueError("Split size must be in (0,100)")
    if not isinstance(input_shape, tuple) and not all((isinstance(elem, int) and elem > 0) for elem in input_shape):
        raise ValueError("Input shape must be a tuple of positive integers.")
    if not isinstance(output_shape, tuple) and not all((isinstance(elem, int) and elem > 0) for elem in output_shape):
        raise ValueError("Output shape must be a tuple of positive integers.")
    if not isinstance(res_cols, list) and not all(isinstance(elem, int) for elem in res_cols):
        raise ValueError("Res cols must be a list of integers.")
    if not isinstance(drop_na, bool):
        raise ValueError("Drop_na must be a boolean value.")
    if not isinstance(col_names, bool):
        raise ValueError("Col_names must be a boolean value.")

def import_module_by_path(module_path, module_name):
    try:
        # remove file extension
        path_without_extension = os.path.splitext(module_path)[0]

        # replace path separators with dots
        path_with_dots = path_without_extension.replace(os.path.sep, '.')
        sys.path.append(module_path)
        return importlib.import_module(f'{module_name}')
        #module = importlib.import_module(f'module_name')
        #spec = importlib.util.spec_from_file_location(module_name, module_path)
        #module = importlib.util.module_from_spec(spec)
        #spec.loader.exec_module(module)
        #return module  
    except Exception as e:
        raise ValueError(f"Failed to import the module {module_name} from {module_path}."\
                         f"Details: {str(e)}")



def gen_model_fitting(model_file, dst_dir, data_file, epochs,
                            batch_size, split_size, input_shape, output_shape, 
                            res_cols, loss, metric, drop_na, col_names):
    
    validate_params(model_file, dst_dir, data_file, epochs,
                    batch_size, split_size, input_shape, output_shape, 
                    res_cols, drop_na, col_names)
    
    dst_model_path = os.path.join(dst_dir, os.path.basename(model_file))

    #копируем модель в итоговую директорию, если ее там нет
    if not os.path.exists(dst_model_path):
        shutil.copyfile(model_file, dst_model_path)

    #проверяем модель
    model_name = os.path.basename(model_file).split('.')[0]
    nn_model = import_module_by_path(dst_dir, model_name)
    model = nn_model.Model()
    
    validate_model(model,input_shape, output_shape)
    
    template_path, loss = get_model_info(model,loss)
    
    # Получаем путь к текущей папке модуля и модели
    module_path = os.path.dirname(os.path.abspath(__file__))
    
    # Формируем путь к файлу шаблона скрипта обучения
    template_path = os.path.join(module_path, template_path)
    train_script_path = os.path.join(dst_dir, 'train_script.py')

    print("..Starting generation of training script...")
    with open(template_path, "r") as source:
        script_template = source.read() 

    with open(train_script_path, "w") as out:
        script_content = script_template.format(
            model_name = model_name, 
            module_path = module_path.replace('\\','/'),
            filename_data = data_file.replace('\\','/'),
            dst_directory = dst_dir.replace('\\','/'),
            epochs = epochs,
            batch_size = batch_size, 
            split_size = split_size,
            input_shape = input_shape, 
            output_shape = output_shape,
            res_cols = res_cols, 
            loss_func = loss,
            metric = metric,
            drop_na = drop_na, 
            col_names = col_names)
        out.write(script_content)       
    print("Generation of training script completed.")

    history_filename = model_name +'_history.json'
    history_path = os.path.join(dst_dir,history_filename)

    return train_script_path,history_path



