import numpy as np
import tensorflow as tf

def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

# file_path = './wider_resnet_28_large.txt'
def model_summary(vars, file_path = None): 
    def shape_parameters(shape):
        v = 1
        for s in shape:
            v *= s
        return v
    
    with open(file_path, 'w') as f:
        f.write('_' * 100 + '\n')
        f.write('{:50s} {:20s} {:20s}'.format('Name', 'Shape', 'Param #') + '\n')
        f.write('_' * 100 + '\n')

        model_params = 0
        
        for var in vars:
            shape = var.shape.as_list()
            params = shape_parameters(shape)

            model_params += params

            f.write('{:50s} {:20s} {:20s}'.format(var.name, str(shape), str(params)) + '\n')
            f.write('_' * 100 + '\n')

        million = model_params / 1000000
        if million >= 1:
            million = str(int(million))
        else:
            million = '{:2f}'.format(million)

        f.write('Total Params : {:,}, {}M'.format(model_params, million) + '\n')
        f.write('_' * 100 + '\n')
