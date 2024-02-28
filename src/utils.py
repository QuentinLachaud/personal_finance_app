import pandas as pd
import numpy as np
import datetime

from src.classes import Asset


def project_returns(capital=1000, annual_rate=.05, years=10, annual_contribution=0):
    ''' 
    function taking initial capital, annual rate and investment period (years)
    to project value per year of maturing capital
    '''
    returns = [capital]
    for _ in range(1, years):
        returns.append((returns[-1] + annual_contribution) * (1 + annual_rate))

    returns_df = pd.DataFrame({'total_value': returns})

    return returns_df

def make_net_worth_df(assets, project_years):
    """_summary_

    Args:
        assets (list): list of Asset instances
        project_years (int): how many years to project over

    Returns:
        pd.DataFrame: Table with annual value fo each asset
    """

    # Instantiate empty df
    net_worth_df = pd.DataFrame()

    # Generate dataframe for net worth projections
    for asset in assets:
        asset.nyears = project_years
        net_worth_df = pd.concat([net_worth_df, pd.DataFrame({asset.name: asset.annual_projection(project_years)})], axis=1)

    return net_worth_df


# Other functions ----------------------------------------------

def random_walk(start_point=0, walk_to=100, step=1, df=True):
        #Instantiate start point
        walk = [start_point]

        for i in range(walk_to):
            walk.append(walk[-1] + np.random.choice([step, -step]))
        if not df:
            return walk
        table = pd.DataFrame({'walk': walk})

        return table

def change_text_colour(age=1):
    """_summary_

    Args:
        age (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: returns the color
    """

    lookup = {range(0, 45): 'green',
                range(45, 60): 'orange',
                range(60, 80): 'red',
                range(80, 1000): 'rainbow'}
    
    for time, color in lookup.items():
        if age in time:
            return color
    return ''

def generate_retirement_portfolio(ret, vol, years=10, periods=1, sims=1, investment=1):
    """_summary_

    Args:
        ret (_type_): annual return (as decimal, e.g. 5.0 for 5% return)
        vol (_type_): annual volatility variance (as decimal, e.g. 15.0 for 15% volatility)
        periods (int, optional): Periods per year. Defaults to 1.
        sims (int, optional): simulations. Defaults to 1.
        investment (int, optional): starting amount to project forward. Defaults to 1.

    Returns:
        _type_: dataframe of investment value over n simualtions and m periods
        
    """
    returns_arr = np.random.normal(loc=ret /100 / periods,
                                         scale=vol / 100 / np.sqrt(periods),
                                         size=(periods * years, sims))
    
    returns_arr_cumprod = (1 + returns_arr).cumprod(axis=0)


    daily_returns_df         = pd.DataFrame(returns_arr_cumprod)
    daily_returns_df.columns = [f'sim_{i}' for i in range(sims)]

    cumulative_returns_df = daily_returns_df * investment

    return cumulative_returns_df