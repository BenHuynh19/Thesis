import pandas as pd
import numpy as np
import scipy.stats as sc
from scipy.optimize import minimize
import matplotlib.pyplot as plt

path = 'data/thesis/'

# The codes below are mainly built based upon the course 
# Introduction to Portfolio Construction and Analysis with Python, taught by Vijay Vaidyanathan, Phd

def compound(r):
    return np.expm1(np.log1p(r).sum())          
        
def get_hf_return (file_name):
    df = pd.read_csv(f'{path}{file_name}', header=0, index_col=0, parse_dates=True)
    hfi = df/100
    hfi.index = hfi.index.to_period('M')
    hfi.columns = hfi.columns.str.strip()
    return hfi
           
def get_df(file_name, path=path):
    df = pd.read_excel(f'{path}{file_name}', parse_dates=True, header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y%m').to_period('M')
    df.columns = df.columns.str.strip()
    return df

def get_df_excel (file_name, path=path):
    df = pd.read_excel(f'{path}{file_name}', parse_dates=True, header=0, index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y%m').to_period('M')
    df.columns = df.columns.str.strip()
    return df

def get_hf (file_name, path=path):
    df = pd.read_excel(f'{path}{file_name}', parse_dates=True, header=0, index_col=0)
    df = df/100
    df.index = pd.to_datetime(df.index)
    df.index = df.index.to_period('M')
    #df.columns = df.columns.str.strip()
    return df

# 7 functions below are used for descriptive purpose
def drawdown (return_series: pd.Series):
    wealth = np.multiply(1000, (1 + return_series).cumprod())
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    return pd.DataFrame({'Wealth': wealth, 'Peak': peak, 'Drawdown': drawdown})    

def get_skewness (return_series):
    e = (return_series - return_series.mean())**3
    skewness = e.mean() / ((return_series.std(ddof=0))**3)
    return skewness

def get_kurtosis (return_series):
    de_mean = (return_series - return_series.mean())**4
    kurtosis = de_mean.mean() / ((return_series.std(ddof=0))**4)
    return kurtosis

def is_normal(r, level=0.01):
    stat, p_value = sc.jarque_bera(r)
    return p_value > level

def his_var (r, level=5):
    var = np.percentile(r, level, axis=0)
    ind = r.columns
    df =  pd.DataFrame(var, index=ind, columns=['VaR'])
    return df

def var_historic (r, level=5):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, q=level)
    else:
        raise TypeError('Expected type is Series or DataFrame')
      
def var_gaussian(r, level=5, modified=False):
    z = sc.norm.ppf(level/100)
    if modified:
        s = get_skewness(r)
        k = get_kurtosis(r)
        z = (z + s/6*(z**2 - 1)
              + (k-3)/24*(z**3 - 3*z)
              - (2*(z**3) - 5*z)*(s**2)/36)
  
    var = r.mean() + z*r.std(ddof=0)  
    return -var

def annualize_ret (r, period_per_year=12):
    compounded_growth = (1+r).prod()
    n_period = r.shape[0]
    ann_ret = compounded_growth**(period_per_year/n_period) - 1
    return ann_ret

def annualize_vol (r, period_per_year=12):
    return r.std()*(period_per_year**0.5)

def sharpe_ratio (r, risk_free, period_per_year=12):
    rf_per_period = (1 + risk_free)**(1/period_per_year) - 1
    excess_ret = r - rf_per_period
    ann_excess_ret = annualize_ret(excess_ret, period_per_year=period_per_year)
    ann_vol = annualize_vol(r, period_per_year=period_per_year)
    return ann_excess_ret/ann_vol
                                  
def portfolio_expected_ret (weight, ret):
    return weight.T @ ret

def portfolio_vol (weight, covariance):
    return (weight.T @ covariance @ weight)**0.5

def plot_ef2 (er, cov, n_points=19):
    weight = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    port_ret = [portfolio_expected_ret(w, er) for w in weight]
    port_vol = [portfolio_vol(w, cov) for w in weight]
    ef_2 = pd.DataFrame({'Return': port_ret,
                        'Volatility': port_vol})
    return ef_2.plot.line(x='Volatility', y='Return')

def max_sharpe_port_ret_vol (er, cov, weight, n_points=19):
    port_ret = portfolio_expected_ret(weight, er)
    port_vol = portfolio_vol(weight, cov)
    return port_ret, port_vol

# Minimizing volatility function
def minimize_vol (target_ret, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bnd = ((0.0, 1.0),)*n   # weights range from 0 to 1
    ret_is_target = {
        'type': 'eq',
        'args': (er,),
        'fun': lambda weights, er: target_ret - portfolio_expected_ret(weights, er)
    }
    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    result = minimize(fun=portfolio_vol, x0=init_guess, args=(cov,), # X0 are the weights
                      method='SLSQP', options={'disp': False},
                       constraints=(ret_is_target, weights_sum_to_1),
                       bounds=bnd
                      )
    return result.x
        
def optimal_w (er, cov, n_points=19):
    target_ret = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_ret]
    return weights

def optimal_w_bm (bm_ret, er, cov):
    weights = minimize_vol(bm_ret, er, cov)
    return weights

def portfolio_max_sharpe (riskfree_rate, er, cov):
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bnd = ((0.0, 1.0),)*n

    weights_sum_to_1 = {
        'type': 'eq',
        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe (weights, riskfree_rate, er, cov):
        r = portfolio_expected_ret(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol
    
    result = minimize(fun=neg_sharpe, x0=init_guess, args=(riskfree_rate, er, cov,), 
                      method='SLSQP', options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bnd
                      )
    return result.x        
        
def gmv (cov):   # global minimum variance portfolio weights
    n = cov.shape[0]
    gmv = portfolio_max_sharpe(0, np.repeat(1, n), cov)
    return gmv

def plot_ef (er, cov, n_points=19, riskfree_rate=0.1, title='', show_ew=False, show_gmv=False, show_cml=False):
    weight = optimal_w(er, cov, n_points)
    port_ret = [portfolio_expected_ret(w, er) for w in weight]
    port_vol = [portfolio_vol(w, cov) for w in weight]
    ef = pd.DataFrame({'Return': port_ret,
                        'Volatility': port_vol})
    ax = ef.plot.line(x='Volatility', y='Return', legend=None)
    ax.set_title(f'{title}')
    ax.set_ylabel('Return')
    
    if show_ew:
        n = er.shape[0]
        ew = np.repeat(1/n, n)
        r_ew = portfolio_expected_ret(ew, er)
        vol_ew = portfolio_vol(ew, cov)
        # draw a point of the equally weghted portfolio
        ax.plot([vol_ew], [r_ew], color='goldenrod', marker='o')
    
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_expected_ret(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        ax.plot([vol_gmv], [r_gmv], color='green', marker='o')
        
    if show_cml:
        ax.set_xlim(left=0)
        w_best = portfolio_max_sharpe(riskfree_rate, er, cov)
        r_best = portfolio_expected_ret(w_best, er)
        vol_best = portfolio_vol(w_best, cov)
        # draw a Capital Market Line (cml)
        x_cml = [0, vol_best]
        y_cml = [riskfree_rate, r_best]
        ax.plot(x_cml, y_cml, color='red', marker='o', linestyle='dashed')
        
    return ax


def plot_ef_with_bm (er, cov, bm, n_points=19, riskfree_rate=0.1, title='', show_ef_point=False, show_bm=False):
    weight = optimal_w(er, cov, n_points)
    port_ret = [portfolio_expected_ret(w, er) for w in weight]
    port_vol = [portfolio_vol(w, cov) for w in weight]
    ef = pd.DataFrame({'Return': port_ret,
                        'Volatility': port_vol})
    ax = ef.plot.line(x='Volatility', y='Return', legend=None)
    ax.set_title(f'{title}')
    ax.set_ylabel('Return')
    
    if show_bm:
        ax.set_xlim(0, 0.1)
        ax.set_ylim(0, 0.02)
        r_gmv = bm[0]
        vol_gmv = bm[1]
        ax.plot([vol_gmv], [r_gmv], color='green', marker='o')
    
    if show_ef_point:
        w_ef = optimal_w_bm(bm[0], er, cov)
        r_ef = portfolio_expected_ret(w_ef, er)
        vol_ef = portfolio_vol(w_ef, cov)
        ax.plot([vol_ef], [r_ef], color='red', marker='o')
        
    return ax

def spotting_p (bm, er, cov):
        w_ef = optimal_w_bm(bm[0], er, cov)
        r_ef = portfolio_expected_ret(w_ef, er)
        vol_ef = portfolio_vol(w_ef, cov)
        res = pd.DataFrame({'Weight': w_ef,
                            'Return': r_ef,
                            'Vol': vol_ef})
        return res

def calculating_ef (er, cov, n_points=19, riskfree_rate=0.1, name=''):
    weight = optimal_w(er, cov, n_points)
    port_ret = [portfolio_expected_ret(w, er) for w in weight]
    port_vol = [portfolio_vol(w, cov) for w in weight]
    ef = pd.DataFrame({'Return': port_ret,
                        'Volatility': port_vol,
                        'Approach': name})
    return ef

def port_gmv (er, cov):
    w_gmv = gmv(cov)
    r_gmv = portfolio_expected_ret(w_gmv, er)
    vol_gmv = portfolio_vol(w_gmv, cov)
    return r_gmv, vol_gmv

def port_tangency (riskfree_rate, er, cov):
    w_best = portfolio_max_sharpe(riskfree_rate, er, cov)
    r_best = portfolio_expected_ret(w_best, er)
    vol_best = portfolio_vol(w_best, cov)
    return r_best, vol_best   
        
        
        
        
        
        