import PySimpleGUI as sg
import pandas as pd
import re
import os

from lib.mypckg import confutils
from lib.mypckg.nn_learning_builder import gen_model_fitting
from lib.mypckg.metrics_visualization import visual

def read_table():
    filename = sg.popup_get_file('Выберите файл датасета', no_window=True, file_types=(("Файлы CSV","*.csv"),("Файлы Excel", "*.xlsx .xlx")))
    print(filename)
    if filename == '':
        return
    
    data = []
    header_list = []

    colnames_checked = sg.popup_yes_no('В файле есть названия столбцов?', background_color='#312c4d', no_titlebar=True) == 'Yes'
    dropnan_checked = sg.popup_yes_no('Удалить строки с NaN?', background_color='#312c4d', no_titlebar=True) == 'Yes'

    if filename is not None:
        file_ext = filename.split('/')[-1].split('.')[-1].lower()
        try:                     
            if colnames_checked:
                df = pd.read_csv(filename, nrows=5) if file_ext == "csv" else pd.read_excel(filename, nrows=5)
                header_list = list(df.columns)    
            else:
                df = pd.read_csv(filename, header=None, nrows=5) if file_ext == "csv" else pd.read_excel(filename, header=None, nrows=5)
                header_list = ['col' + str(x) for x in range(len(df.iloc[0]))]
                df.columns = header_list
            if dropnan_checked:
                df = df.dropna()
            data = df.to_numpy().tolist()
            return (data, header_list,filename,dropnan_checked, colnames_checked)
        except:
            sg.popup_error('Ошибка чтения файла')
            return

def show_table(data, header_list, fn):  
    layout = [[sg.Push(), sg.Table(values=data[:5], headings=header_list,
                  font='Calibri', pad=(25,25), display_row_numbers=False, 
                  vertical_scroll_only=False, auto_size_columns = True,
                  hide_vertical_scroll=True, num_rows=min(5,len(data[:5]))), sg.Push()],]

    window = sg.Window(fn, layout, grab_anywhere=True, resizable=False, size=(1000,250))
    while True:
        event, values = window.read(timeout=1)
        if event in (sg.WIN_CLOSED, 'Exit'):
            break
    window.close()


def run_file(pyfile, window=None):
    p = sg.execute_py_file(pyfile,pipe_output=True)
    output = ''
    while True:
        line = p.stdout.readline()
        if not line:
            break
        line = line.decode(errors='backslashreplace').rstrip()
        output += line
        print(line)
        if window:
            window.Refresh()

    retval = p.wait()
    return (retval, output)
#=====================================================#
loss_functions = [
    'mean_squared_error',
    'mean_absolute_error',
    'categorical_crossentropy'
]
metrics= [
    'acc','mae','mse'
]
# Define the window's contents i.e. layout
fr_train_layout =[[sg.Text('Модель: ', expand_x=True, font='Corbel 12', pad=(5,5)), sg.Input(disabled = True), 
                   sg.FileBrowse('Выбрать', font='Corbel 12', file_types=(("Файлы Python", "*.py"),), key='modelfile')],
    [sg.Text('Файл конфигурации:', font='Corbel 12', pad=(5,5)), sg.Input(disabled = True), 
        sg.FileSaveAs('Выбрать', font='Corbel 12', file_types=(("INI-файлы", "*.ini"),), key='configfile')],
    [sg.Text('Количество эпох: ', font='Corbel 12', pad=(5,5)), 
        sg.Input(size=(10,1), enable_events=True, key='-epochs-')],
    [sg.Text('Размер батча: ', font='Corbel 12', pad=(5,5)), 
         sg.Input(size=(10,1), enable_events=True, key='-batch-')],
    [sg.Text('Размер обучающей выборки в процентах: ', font='Corbel 12', pad=(5,5)), 
        sg.Input(size=(10,1), enable_events=True, key='-split_size-')],
    [sg.Text('Форма входных данных: ', font='Corbel 12', pad=(5,5)), 
        sg.Input(size=(10,1), enable_events=True, key='-inp_shape-')],
    [sg.Text('Форма выходных данных: ', font='Corbel 12', pad=(5,5)), 
        sg.Input(size=(10,1), enable_events=True, key='-out_shape-')],
    [sg.Text('Функция потерь:', font='Corbel 12', pad=(5,5)),
        sg.DropDown(loss_functions, key='loss_func', size=(30, 1), readonly=True)],
    [sg.Text('Метрика:', font='Corbel 12', pad=(5,5)),
        sg.DropDown(metrics, key='metric', size=(30, 1), readonly=True)]]

fr_dataset_layout = [[sg.Button('Загрузить данные', enable_events=True, font='Corbel 12', key='-LOAD-'),
        sg.Button('Показать данные',size=(16,1), enable_events=True, font='Corbel 12', key='-SHOW-')], 
        [sg.Text("", size=(50,1), pad=(5,1), font='Corbel 12', key='-dataset_info-'),],
        [sg.Text("Выберите выходные столбцы:",size=(25,1), pad=(5,5), font='Calibri 12'),],    
        [sg.Listbox(values=(''),size=(37,6), select_mode = 'extended', key='-colnames-'),],
        [sg.Button('Выбрать',size=(15,1),enable_events=True, font='Corbel 12', key='-SAVE-'),]]

layout = [[sg.Frame('Датасет',fr_dataset_layout,size=(350,310),font='Corbel 12'),
                sg.Frame('Параметры обучения', fr_train_layout, font='Corbel 12')],
        [sg.Button('Сохранить параметры', font='Corbel 12', pad=(5,5), key='-SAVE_PARAMS-'),
         sg.Button('Начать обучение', font='Corbel 12', pad=(5,5), key='-START-FIT-'),
         sg.Button('Открыть график обучения', font='Corbel 12', pad=(5,5), key='-PLOT-')],
        [sg.Output(size=(65, 15), font='Courier 10', key='-log-',expand_x=True,expand_y=True)]]

# Create the window
window = sg.Window('Подсистема обучения моделей НС', layout, size=(1100,600), resizable=True)

output_columns_indices = []
filename_ds = ''
filename_model = ''
filename_config = ''
dst = ''
filename_history =''
read_successful = False
drop_na = True
col_names = True
metric =''
# Event loop
while True:
    event, values = window.read()
    dataset_info = window['-dataset_info-']
    if event in (sg.WIN_CLOSED, 'Exit'):
        break

    if event == '-LOAD-':
        try:
            data, header_list, filename_ds, drop_na, col_names = read_table()
            read_successful = True
        except:
            pass
        if read_successful:
            fn = filename_ds.split('/')[-1]
            output_columns_indices = []
            dataset_info.update("Датасет: '{}'".format(fn), text_color='white')
            window.Element('-colnames-').update(values=header_list,)

    if event == '-SHOW-':
        if read_successful:
            show_table(data,header_list,fn)
        else:
            dataset_info.update("Датасет не был выбран", text_color='maroon')

    if event == '-PLOT-':
        if not os.path.exists(filename_history):
            print(f"Не удалось найти файл с историей обучения: {filename_history}") 
        else:
            try:
                visual(filename_history,metric)
            except Exception as e:
                print("Failed to visualize.")
                print("Details:", str(e))
                sg.PopupError('Не удалось визуализировать данные')

    if event in ('-epochs-','-batch-','-split_size-'):
        if not values[event].isdigit():
            sg.popup_quick_message('Ошибка: значение должно быть неотрицательным целым числом.', background_color='maroon')
            new_val = ''
            window[event].update(new_val)

    if event in ('-inp_shape-','-out_shape-'):
        if values[event] != '': 
            if not values[event][-1].isdigit() and values[event][-1] != ',':
                sg.popup_quick_message('Ошибка: некорректный ввод.', background_color='maroon')
                new_val = ''
                window[event].update(new_val)
        
    if event == '-SAVE_PARAMS-':
        pattern = r'^\d+(,\d+)*$'
        epochs, batch_size, split_size = values['-epochs-'], values['-batch-'], values['-split_size-']
        input_shape, output_shape = values['-inp_shape-'], values['-out_shape-']
        filename_config = values['configfile']
        filename_model = values['modelfile']
        loss_func =  values['loss_func']
        metric = values['metric']
        params = (epochs, batch_size, split_size, filename_config, input_shape, output_shape, loss_func, metric)
        if not all([bool(s) for s in params]):
            sg.popup_quick_message('Ошибка: заполните значения параметров обучения.', background_color='maroon')
        elif not bool(output_columns_indices):
            sg.popup_quick_message('Ошибка: выберите датасет и выходные столбцы.', background_color='maroon')
        elif not os.path.exists(filename_model):
            sg.popup_quick_message('Ошибка: файл модели не существует.', background_color='maroon')
        elif re.match(pattern, input_shape) and re.match(pattern, output_shape):
            confutils.save_fit_param(filename_config,filename_model,filename_ds,
                        epochs, batch_size, split_size, 
                        input_shape, output_shape, 
                        output_columns_indices, loss_func, metric,
                        drop_na, col_names)
            sg.popup_quick_message('Параметры сохранены.', background_color='darkcyan')
        else:
            sg.popup_quick_message('Ошибка: входная и(-или) выходная форма некоретна(-ы).', background_color='maroon')

    if event=='-START-FIT-':
        try:
            filename_config = values['configfile']
            if not os.path.exists(filename_config):
                raise ValueError(f"Не удалось найти конфигурационный файл {filename_config}") 
            p = confutils.get_fit_param(filename_config)
            metric = p['metric']
            dst = sg.popup_get_folder('Выберите папку', title='Выберите директорию для сохранения', no_window=True,)
            print("Выбранная директория: " + str(dst))
            if dst =='':
                raise ValueError("Не была выбрана директория для сохранения")
            path,filename_history = gen_model_fitting(p['model_path'], os.path.normpath(dst),  p['dataset_path'],
                                p['epochs'], p['batch_size'], p['split_size'], 
                                p['input_shape'], p['output_shape'], p['res_cols'], 
                                p['loss'], p['metric'], p['drop_na'], p['col_names']) 
            print('Making ...the program has NOT locked up...')
            window.refresh()
            out, err = run_file(window=window, pyfile=path)
            print('**** DONE **** \n')
            with open(os.path.join(dst,"log.txt"), "w") as text_file:
                text_file.write(window['-log-'].get())
        except Exception as e:
            print("Failed to train model.")
            print("Details:", str(e))
            sg.PopupError('Обучение прервано')

    if event=='-SAVE-':
        print("Выбранные столбцы: " + str(values['-colnames-']))
        colnames_list = window['-colnames-'].GetListValues()
        if len(colnames_list) != 0:
            output_columns_indices = [i for i, name in enumerate(colnames_list) if name in values['-colnames-']]
            print("Индексы выбранных столбцов: "+ str(output_columns_indices))
            if len(output_columns_indices) == 0:
                sg.popup_quick_message('Ошибка: выходные столбцы не выбраны.', background_color='maroon')
        else:
            sg.popup_quick_message('Ошибка: исходный список пуст.', background_color='maroon')

