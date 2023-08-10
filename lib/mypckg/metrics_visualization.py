import json
import matplotlib.pyplot as plt

def one_plot(n, y_lb, loss, val_loss,label = ''): 
    plt.subplot(1, 2, n) 
    if n == 1: 
        lb, lb2 = 'loss', 'val_loss' 
        yMin = 0 
        yMax = 1.05 * max(max(loss), max(val_loss)) 
    else: 
        lb, lb2 = label, 'val_'+label 
        yMin = min(min(loss), min(val_loss)) 
        yMax = 1.05 * max(max(loss), max(val_loss)) 
    plt.plot(loss, color = 'r', label = lb, linestyle = '--') 
    plt.plot(val_loss, color = 'g', label = lb2) 
    plt.ylabel(y_lb) 
    plt.xlabel('Эпоха') 
    plt.ylim([0.95 * yMin, yMax]) 
    plt.legend() 

def visual(history_path,metric):
    with open(history_path, 'r') as f:
        history_dict = json.load(f)
    suptitle = 'Потери и '+ metric
    if 'loss' in history_dict and 'val_loss' in history_dict:
        plt.figure(figsize = (9, 4)) 
        plt.subplots_adjust(wspace = 0.5) 
        one_plot(1, 'Потери', history_dict['loss'], history_dict['val_loss']) 
        one_plot(2, 'Метрика '+ metric, history_dict[metric], history_dict['val_'+ metric],metric) 
        plt.suptitle(suptitle) 
        plt.show() 