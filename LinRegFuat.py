import pandas as pd
import numpy as np

def linear_regression_fuat(x, y):
    x = pd.Series(x)
    y = pd.Series(y)
    if len(x) != len(y):
        raise ValueError('All X values must pair with a Y value.')
    
    for i in range(len(x)):
        if x[i] == np.NaN:
            raise ValueError('Drop NA Values first with pandas dropNA function.')
        if y[i] == np.NaN:
            raise ValueError('Drop NA Values first with pandas dropNA function.')
    

    
    n = np.size(x) 
    m_x, m_y = np.mean(x), np.mean(y) 
    SS_xy = np.sum(y*x) - n*m_y*m_x 
    SS_xx = np.sum(x*x) - n*m_x*m_x 

    b_1=SS_xy/SS_xx
    b_0 = m_y - b_1*m_x
    y_pred = b_0 + b_1*x	
    
    y_pred_diff = (y - y_pred)**2
    standard_error = np.sqrt(y_pred_diff.sum() / len(x))
    
    
    
    return (y_pred, b_0, b_1, y_pred_diff, standard_error)
    

