# -*- coding: utf-8 -*-
"""
# **Import Libraries**
"""

# Import libraries
from PIL import Image, ImageTk
from tkinter.scrolledtext import ScrolledText
import datetime
import tkinter as tk
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import pandas as pd
import numpy as np
import cplex
import matplotlib.pyplot as plt
import random
import cyipopt as ipopt
import yfinance as yf

"""# Data of Adjusted Closing Prices"""
# Select the asset pool and download the data required.
ticket_100 = ['XLI', 'XLF', 'SPY', 'IVV', 'VOO', 'SPLG', 'SCHX', 'IVW', 'RSP', 'ARKK', 'QQQ', 'VCSH', 'DVY', 'ACWI', 'EMB', 'SCZ', 'VONG', 'RDVY', 'SOXX', 'SMH',  # ETF
              'TLT', 'VCIT', 'JNK', 'HYS', 'BNDX', 'AGG', 'VTIP', 'IGIB', 'USIG', 'VCLT', 'ANGL', 'FALN', 'HYLS', 'SUSC', 'BSJM', 'BSJN', 'VTC', 'BSJO', 'BSJP', 'GRNB',  # bond
              'PYPL', 'C', 'GS', 'VOYA', 'BX', 'SBNY', 'RY', 'BMO', 'TD', 'JPM', 'BAC', 'MA', 'CIHKY', 'CICHY', 'WFC',  # finance
              'NVCR', 'AAPL', 'MSFT', 'AMZN', 'META', 'GOOG', 'NVDA', 'CSCO', 'ORCL', 'ACN', 'ADBE', 'CRM', 'INTC', 'QCOM', 'SHOP',  # technology
              'JNJ', 'PFE', 'TMO', 'ATRI', 'IDXX', 'COO', 'NVS', 'KZR', 'ZTS', 'DHR', 'CNC', 'MRNA', 'BHC', 'MDT', 'INCY',  # healthcare
              'WM', 'LPX', 'BLDR', 'SEB', 'TRU', 'CPA', 'NOC', 'PWR', 'RHI', 'MMM', 'CSX', 'RCRRF', 'CNI', 'RCRUY', 'CP']  # industrial
yf.download(tickers=ticket_100, start='2019-04-08', interval="1d",
            end='2022-04-09')['Adj Close'].to_csv('Daily_closing_prices_100.csv')


"""# Investment Strategies"""

# choose corresponding strategy for different risk tolerance level


def change_portfolio_v2(risk_tol_level, ESG_option, fund_avl, inv_horizon, sect_pref):

    # 20 assets in portfolio
    x_init = [0]*20

    # convert the risk_tol_level to string
    risk_tol_list = ['aggressive', 'slightly aggressive',
                     'moderate', 'slightly conservative', 'conservative']
    risk_bin = [0.6, 1.2, 1.8, 2.4, 3.]
    portfolio_name = risk_tol_list[np.digitize(risk_tol_level, risk_bin)]

    if portfolio_name == 'conservative':  # min. variance

        port_table, Q, mu, cur_prices = filter_asset_pool(
            inv_horizon, ESG_option, sect_pref)
        number_stock, cash_optimal, weight_optimal = strat_min_variance(
            x_init, fund_avl, mu, Q, cur_prices)
        name_stock = port_table.columns.values
        return name_stock, number_stock

    elif portfolio_name == 'slightly conservative':  # equal risk contribution

        port_table, Q, mu, cur_prices = filter_asset_pool(
            inv_horizon, ESG_option, sect_pref)
        number_stock, cash_optimal, weight_optimal = strat_equal_risk_contr(
            x_init, fund_avl, mu, Q, cur_prices)
        name_stock = port_table.columns.values
        return name_stock, number_stock

    elif portfolio_name == 'moderate':  # robust mean-variance

        port_table, Q, mu, cur_prices = filter_asset_pool(
            inv_horizon, ESG_option, sect_pref)
        number_stock, cash_optimal, weight_optimal = strat_robust_optim(
            x_init, fund_avl, mu, Q, cur_prices)
        name_stock = port_table.columns.values
        return name_stock, number_stock

    elif portfolio_name == 'slightly aggressive':  # max. sharp ratio

        port_table, Q, mu, cur_prices = filter_asset_pool(
            inv_horizon, ESG_option, sect_pref)
        number_stock, cash_optimal, weight_optimal = strat_max_Sharpe(
            x_init, fund_avl, mu, Q, cur_prices)
        name_stock = port_table.columns.values
        return name_stock, number_stock

    else:  # leverage max. sharp ratio

        port_table, Q, mu, cur_prices = filter_asset_pool(
            inv_horizon, ESG_option, sect_pref)
        number_stock, cash_optimal, weight_optimal = strat_max_Sharpe(
            x_init, fund_avl, mu, Q, cur_prices)
        number_stock = number_stock*2
        name_stock = port_table.columns.values
        return name_stock, number_stock


# strategy functions

def strat_min_variance(x_init, cash_init, mu, Q, cur_prices):
    N = 20
    # initialize the CPLEX Object
    cpx = cplex.Cplex()
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    # Define linear part of objective
    # function and bounds on variables
    c = [0.0] * N
    ub = [1.0] * N
    # Define constraint matrix A
    cols = []
    for k in range(N):
        cols.append([[0], [1.0]])
    # Add objective function, bounds on variables and constraints to CPLEX model
    cpx.linear_constraints.add(rhs=[1.0], senses="E")
    cpx.variables.add(obj=c, ub=ub, columns=cols, names=[
                      "w_%s" % i for i in range(1, N+1)])
    # Define and add quadratic part of objective function
    Qmat = [[list(range(N)), list(2*Q[k, :])] for k in range(N)]
    cpx.objective.set_quadratic(Qmat)
    # Set CPLEX parameters
    alg = cpx.parameters.lpmethod.values
    cpx.parameters.qpmethod.set(alg.concurrent)
    # Optimize the problem
    cpx.solve()

    # get x_optimal
    x_ratio = np.array(cpx.solution.get_values())
    p_current = np.dot(cur_prices, x_init)+cash_init
    v_current = p_current * x_ratio
    x_optimal = np.floor(v_current / cur_prices)  # rounding down strategy

    # calculate V_sell
    x_sell = x_init - x_optimal
    x_sell[x_sell < 0] = 0
    V_sell = np.dot(cur_prices, x_sell)
    # calculate V_buy
    x_buy = x_optimal - x_init
    x_buy[x_buy < 0] = 0
    V_buy = np.dot(cur_prices, x_buy)
    # calculate transaction cost
    x_change = np.abs(x_init - x_optimal)
    TC = 0.005 * np.dot(cur_prices, x_change)

    # calculate cash_optimal
    cash_optimal = V_sell - V_buy - TC + cash_init

    weight_optimal = x_optimal * cur_prices / \
        np.dot(cur_prices, x_optimal)  # calculate asset weights

    return x_optimal, cash_optimal, weight_optimal


def strat_max_Sharpe(x_init, cash_init, mu, Q, cur_prices):
    r_rf = 0.025
    N = 20
    # initialize the CPLEX Object
    cpx = cplex.Cplex()
    cpx.set_results_stream(None)
    cpx.set_warning_stream(None)
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    # Define linear part of objective
    # function and bounds on variables
    c = [0.0] * N
    ub = [cplex.infinity] * N
    # Define constraint matrix A
    cols = []
    for k in range(N):
        # convert to daily risk free rate
        cols.append([[0], [mu[k]-1/252*r_rf]])
    # Add objective function, bounds on variables and constraints to CPLEX model
    cpx.linear_constraints.add(rhs=[1.0], senses="E")
    cpx.variables.add(obj=c, ub=ub, columns=cols, names=[
                      "y_%s" % i for i in range(1, N+1)])
    # Define and add quadratic part of objective function
    Qmat = [[list(range(N)), list(2*Q[k, :])] for k in range(N)]
    cpx.objective.set_quadratic(Qmat)
    # Set CPLEX parameters
    alg = cpx.parameters.lpmethod.values
    cpx.parameters.qpmethod.set(alg.concurrent)
    # Optimize the problem
    cpx.solve()

    if cpx.solution.get_status_string() == 'infeasible':
        x_optimal = x_init
        cash_optimal = cash_init
        weight_optimal = x_optimal * cur_prices / np.dot(cur_prices, x_optimal)
        return x_optimal, cash_optimal, weight_optimal
    else:
        # get x_optimal
        y = np.array(cpx.solution.get_values())
        x_ratio = y/np.sum(y)
        p_current = np.dot(cur_prices, x_init)+cash_init
        v_current = p_current * x_ratio
        x_optimal = np.floor(v_current / cur_prices)  # rounding down strategy

        # calculate V_sell
        x_sell = x_init - x_optimal
        x_sell[x_sell < 0] = 0
        V_sell = np.dot(cur_prices, x_sell)
        # calculate V_buy
        x_buy = x_optimal - x_init
        x_buy[x_buy < 0] = 0
        V_buy = np.dot(cur_prices, x_buy)
        # calculate transaction cost
        x_change = np.abs(x_init - x_optimal)
        TC = 0.005 * np.dot(cur_prices, x_change)

        # calculate cash_optimal
        cash_optimal = V_sell - V_buy - TC + cash_init

        weight_optimal = x_optimal * cur_prices / \
            np.dot(cur_prices, x_optimal)  # calculate asset weights

        return x_optimal, cash_optimal, weight_optimal


def strat_equal_risk_contr(x_init, cash_init, mu, Q, cur_prices):
    n = 20
    # define the class for ERC

    class erc(object):
        def __init__(self):
            pass

        def objective(self, x):
            y = x * np.dot(Q, x)
            fval = 0
            for i in range(n):
                for j in range(i, n):
                    xij = y[i] - y[j]
                    fval = fval + xij*xij
            fval = 2*fval
            return fval

        def gradient(self, x):
            grad = np.zeros(n)
            t = (np.dot(Q, x)).flatten()
            y = x * t
            for k in range(n):
                g1 = n * t[k] * y[k] - t[k] * np.sum(y)
                g2 = n * np.sum((x * y) * Q[k, :]) - t[k] * np.sum(y)
                grad[k] = 4*(g1+g2)
            return grad

        def constraints(self, x):
            return [1.0] * n

        def jacobian(self, x):
            return np.array([[1.0] * n])

    # Use wo(1/n) as initial portfolio for starting IPOPT optimization

    w0 = [1.0/20] * n
    lb = [0.0] * n  # lower bounds on variables
    ub = [1.0] * n  # upper bounds on variables
    cl = [1]        # lower bounds on constraints
    cu = [1]        # upper bounds on constraints
    Q = Q

    # Define IPOPT problem
    nlp = ipopt.Problem(n=len(w0), m=len(
        cl), problem_obj=erc(), lb=lb, ub=ub, cl=cl, cu=cu)

    # Set the IPOPT options
    nlp.add_option('jac_c_constant'.encode('utf-8'), 'yes'.encode('utf-8'))
    nlp.add_option('hessian_approximation'.encode(
        'utf-8'), 'limited-memory'.encode('utf-8'))
    nlp.add_option('mu_strategy'.encode('utf-8'), 'adaptive'.encode('utf-8'))
    nlp.add_option('tol'.encode('utf-8'), 1e-10)
    nlp.add_option('sb', 'yes')
    nlp.add_option('print_level', 0)

    # Solve the problem
    w_erc, info = nlp.solve(w0)

    # calculate x_optimal
    x_ratio = w_erc/np.sum(w_erc)
    p_current = np.dot(cur_prices, x_init)+cash_init
    v_current = p_current * x_ratio
    x_optimal = np.floor(v_current / cur_prices)  # rounding down strategy

    # calculate V_sell
    x_sell = x_init - x_optimal
    x_sell[x_sell < 0] = 0
    V_sell = np.dot(cur_prices, x_sell)
    # calculate V_buy
    x_buy = x_optimal - x_init
    x_buy[x_buy < 0] = 0
    V_buy = np.dot(cur_prices, x_buy)
    # calculate transaction cost
    x_change = np.abs(x_init - x_optimal)
    TC = 0.005 * np.dot(cur_prices, x_change)

    # calculate cash_optimal
    cash_optimal = V_sell - V_buy - TC + cash_init

    weight_optimal = x_optimal * cur_prices / \
        np.dot(cur_prices, x_optimal)  # calculate asset weights

    return x_optimal, cash_optimal, weight_optimal


def strat_robust_optim(x_init, cash_init, mu, Q, cur_prices):
    n = 20
    # Initial portfolio ("equally weighted" or "1/n")
    w0 = [1.0/n] * n

    # Target portfolio return estimation error
    var_matr = np.diag(np.diag(Q))
    rob_init = np.dot(w0, np.dot(var_matr, w0))  # r.est.err of 1/n port
    rob_bnd = rob_init  # target return estimation error

    # compute the minimum variance
    w_minVar = strat_min_variance(x_init, cash_init, mu, Q, cur_prices)[2]
    ret_minVar = np.dot(mu, w_minVar)
    # Target portfolio return is return of minimum variance portfolio
    Portf_Retn = ret_minVar

    # Formulate and solve robust mean variance problem
    cpx = cplex.Cplex()
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    c = [0.0] * n
    lb = [0.0] * n
    ub = [1.0] * n
    A = []
    for k in range(n):
        A.append([[0, 1], [1.0, mu[k]]])
    var_names = ["w_%s" % i for i in range(1, n+1)]
    cpx.linear_constraints.add(rhs=[1.0, Portf_Retn], senses="EG")
    cpx.variables.add(obj=c, lb=lb, ub=ub, columns=A, names=var_names)
    Qmat = [[list(range(n)), list(2*Q[k, :])] for k in range(n)]
    cpx.objective.set_quadratic(Qmat)
    Qcon = cplex.SparseTriple(
        ind1=var_names, ind2=range(n), val=np.diag(var_matr))
    cpx.quadratic_constraints.add(rhs=rob_bnd, quad_expr=Qcon, name="Qc")
    cpx.parameters.threads.set(4)
    #print("Setting number of threads = ", 4)
    cpx.parameters.timelimit.set(60)
    #print("Setting timelimit = ", 60)
    cpx.parameters.barrier.qcpconvergetol.set(1e-12)
    #print("Setting Barrier algorithm convergence tolerance = ", 1e-12)
    cpx.set_results_stream(None)
    cpx.solve()

    weight = cpx.solution.get_values()
    w_ro = np.array(weight)

    # calculate x_optimal
    x_ratio = w_ro/np.sum(w_ro)
    p_current = np.dot(cur_prices, x_init)+cash_init
    v_current = p_current * x_ratio
    x_optimal = np.floor(v_current / cur_prices)  # rounding down strategy

    # calculate V_sell
    x_sell = x_init - x_optimal
    x_sell[x_sell < 0] = 0
    V_sell = np.dot(cur_prices, x_sell)
    # calculate V_buy
    x_buy = x_optimal - x_init
    x_buy[x_buy < 0] = 0
    V_buy = np.dot(cur_prices, x_buy)
    # calculate transaction cost
    x_change = np.abs(x_init - x_optimal)
    TC = 0.005 * np.dot(cur_prices, x_change)

    # calculate cash_optimal
    cash_optimal = V_sell - V_buy - TC + cash_init

    weight_optimal = x_optimal * cur_prices / \
        np.dot(cur_prices, x_optimal)  # calculate asset weights

    return x_optimal, cash_optimal, weight_optimal


"""# **Additional Functions**"""

# ETF & bond & stock pool selection
# Filter the asset pool based on invester's preference


def filter_asset_pool(inv_horizon, ESG, stock_sector):

    # If the time horizon is longer than 2 years,
    long_time_horizon = (inv_horizon >= 2)

    df_price = pd.read_csv('Daily_closing_prices_100.csv')

    df_detail = pd.read_excel('Asset_Pool_Detail.xlsx')

    df1 = df_detail[(df_detail['Type'] == 'ETF') |
                    (df_detail['Type'] == 'bond')]
    df2 = df_detail[df_detail['Type'] == 'stock']

    if long_time_horizon == False:  # short-term: randomly choose 10 etfs and 10 bonds

        select_tick_etf = df1.loc[(df1['Type'] == 'ETF'), 'Symbol'].tolist()
        select_tick_etf = random.choices(select_tick_etf, k=10)

        select_tick_bond = df1.loc[(df1['Type'] == 'bond'), 'Symbol'].tolist()
        select_tick_bond = random.choices(select_tick_bond, k=10)

        select_tick = select_tick_etf + select_tick_bond

    elif long_time_horizon == True:  # long-term: randomly choose 10 stocks from that sector, 5 etfs, and 5 bonds

        if ESG == 1:
            df2 = df2[df2['ESG'] == 'Yes']

        if stock_sector == 1:
            df2 = df2[df2['Sector'] == 'technology']

        if stock_sector == 2:
            df2 = df2[df2['Sector'] == 'healthcare']

        if stock_sector == 3:
            df2 = df2[df2['Sector'] == 'finance']

        if stock_sector == 4:
            df2 = df2[df2['Sector'] == 'industrial']

        # select 10 stocks, 5 etfs, and 5 bonds

        select_tick_etf = df1.loc[(df1['Type'] == 'ETF'), 'Symbol'].tolist()
        select_tick_etf = random.choices(select_tick_etf, k=5)

        select_tick_bond = df1.loc[(df1['Type'] == 'bond'), 'Symbol'].tolist()
        select_tick_bond = random.choices(select_tick_bond, k=5)

        select_tick_stock = df2.loc[(
            df2['Type'] == 'stock'), 'Symbol'].tolist()
        select_tick_stock = random.choices(select_tick_stock, k=10)

        select_tick = select_tick_etf + select_tick_bond + select_tick_stock

    port_table = df_price[select_tick]

    # return Q, mu, current price - port_table contains the 759 rows x the total number of asset==20
    ret = port_table.pct_change().iloc[1:, :]
    ret.reset_index(inplace=True, drop=True)
    # Compute the covariance matrix of the percent change of prices
    Q = ret.cov().to_numpy()
    # Compute the mean for each column
    mu = np.array(np.mean(ret, axis=0))
    # return the current price from the last row
    price = port_table.iloc[-1, :]

    return port_table, Q, mu, price

# portfolio daily value for past year (2021) assuming initial fund available = 10000


def portfolio_daily_value_2021(inv_horizon, ESG, stock_sector, risk_tol_level):

    x_init = [0]*20
    fund_avl = 10000
    perf_daily_value = []
    port_table = filter_asset_pool(inv_horizon, ESG, stock_sector)[0]
    # convert the risk_tol_level to string
    risk_tol_list = ['aggressive', 'slightly aggressive',
                     'moderate', 'slightly conservative', 'conservative']
    risk_bin = [0.6, 1.2, 1.8, 2.4, 3.]
    portfolio_name = risk_tol_list[np.digitize(risk_tol_level, risk_bin)]

    if portfolio_name == 'conservative':  # min. variance

        # reposition every 2 months, first reposition is January 2021; last reposition is November 2021
        for period in range(1, 7):
            # calculate mu and Q from previous 2 month data
            # return Q, mu, current price - port_table contains the 759 rows x the total number of asset==20
            # 398 is the index for 2021/11/2; +42 is 2 months
            index = 398+42*(period-1)
            ret = port_table.pct_change().iloc[index:(index+42), :]
            ret.reset_index(inplace=True, drop=True)
            # Compute the covariance matrix of the percent change of prices
            Q = ret.cov().to_numpy()
            # Compute the mean for each column
            mu = np.array(np.mean(ret, axis=0))
            # return the current price from the last row
            cur_prices = port_table.iloc[(index+42), :]

            # calculate portfolio reposition
            number_stock, cash_optimal, weight_optimal = strat_min_variance(
                x_init, fund_avl, mu, Q, cur_prices)
            x_init = number_stock
            fund_avl = cash_optimal

            # calculate daily portfolio value and append to perf_daily_value
            for i in range(42):
                perf_daily_value.append(
                    np.dot(number_stock, port_table.iloc[(index+42+i), :])+cash_optimal)

        return perf_daily_value

    elif portfolio_name == 'slightly conservative':  # equal risk contribution

        # reposition every 2 months, first reposition is January 2021; last reposition is November 2021
        for period in range(1, 7):
            # calculate mu and Q from previous 2 month data
            # return Q, mu, current price - port_table contains the 759 rows x the total number of asset==20
            # 398 is the index for 2021/11/2; +42 is 2 months
            index = 398+42*(period-1)
            ret = port_table.pct_change().iloc[index:(index+42), :]
            ret.reset_index(inplace=True, drop=True)
            # Compute the covariance matrix of the percent change of prices
            Q = ret.cov().to_numpy()
            # Compute the mean for each column
            mu = np.array(np.mean(ret, axis=0))
            # return the current price from the last row
            cur_prices = port_table.iloc[(index+42), :]

            # calculate portfolio reposition
            number_stock, cash_optimal, weight_optimal = strat_equal_risk_contr(
                x_init, fund_avl, mu, Q, cur_prices)
            x_init = number_stock
            fund_avl = cash_optimal

            # calculate daily portfolio value and append to perf_daily_value
            for i in range(42):
                perf_daily_value.append(
                    np.dot(number_stock, port_table.iloc[(index+42+i), :])+cash_optimal)

        return perf_daily_value

    elif portfolio_name == 'moderate':  # robust mean-variance

        # reposition every 2 months, first reposition is January 2021; last reposition is November 2021
        for period in range(1, 7):
            # calculate mu and Q from previous 2 month data
            # return Q, mu, current price - port_table contains the 759 rows x the total number of asset==20
            # 398 is the index for 2021/11/2; +42 is 2 months
            index = 398+42*(period-1)
            ret = port_table.pct_change().iloc[index:(index+42), :]
            ret.reset_index(inplace=True, drop=True)
            # Compute the covariance matrix of the percent change of prices
            Q = ret.cov().to_numpy()
            # Compute the mean for each column
            mu = np.array(np.mean(ret, axis=0))
            # return the current price from the last row
            cur_prices = port_table.iloc[(index+42), :]

            # calculate portfolio reposition
            number_stock, cash_optimal, weight_optimal = strat_robust_optim(
                x_init, fund_avl, mu, Q, cur_prices)
            x_init = number_stock
            fund_avl = cash_optimal

            # calculate daily portfolio value and append to perf_daily_value
            for i in range(42):
                perf_daily_value.append(
                    np.dot(number_stock, port_table.iloc[(index+42+i), :])+cash_optimal)

        return perf_daily_value

    elif portfolio_name == 'slightly aggressive':  # max. sharp ratio

        # reposition every 2 months, first reposition is January 2021; last reposition is November 2021
        for period in range(1, 7):
            # calculate mu and Q from previous 2 month data
            # return Q, mu, current price - port_table contains the 759 rows x the total number of asset==20
            # 398 is the index for 2021/11/2; +42 is 2 months
            index = 398+42*(period-1)
            ret = port_table.pct_change().iloc[index:(index+42), :]
            ret.reset_index(inplace=True, drop=True)
            # Compute the covariance matrix of the percent change of prices
            Q = ret.cov().to_numpy()
            # Compute the mean for each column
            mu = np.array(np.mean(ret, axis=0))
            # return the current price from the last row
            cur_prices = port_table.iloc[(index+42), :]

            # calculate portfolio reposition
            number_stock, cash_optimal, weight_optimal = strat_max_Sharpe(
                x_init, fund_avl, mu, Q, cur_prices)
            x_init = number_stock
            fund_avl = cash_optimal

            # calculate daily portfolio value and append to perf_daily_value
            for i in range(42):
                perf_daily_value.append(
                    np.dot(number_stock, port_table.iloc[(index+42+i), :])+cash_optimal)

        return perf_daily_value

    else:  # leverage max. sharp ratio

        # reposition every 2 months, first reposition is January 2021; last reposition is November 2021
        for period in range(1, 7):
            # calculate mu and Q from previous 2 month data
            # return Q, mu, current price - port_table contains the 759 rows x the total number of asset==20
            # 398 is the index for 2021/11/2; +42 is 2 months
            index = 398+42*(period-1)
            ret = port_table.pct_change().iloc[index:(index+42), :]
            ret.reset_index(inplace=True, drop=True)
            # Compute the covariance matrix of the percent change of prices
            Q = ret.cov().to_numpy()
            # Compute the mean for each column
            mu = np.array(np.mean(ret, axis=0))
            # return the current price from the last row
            cur_prices = port_table.iloc[(index+42), :]

            # calculate portfolio reposition
            number_stock, cash_optimal, weight_optimal = strat_max_Sharpe(
                x_init, fund_avl, mu, Q, cur_prices)
            if period == 1:
                fund_borrowed = np.dot(number_stock, cur_prices)
            number_stock = number_stock*2
            x_init = number_stock/2
            fund_avl = cash_optimal

            # calculate daily portfolio value and append to perf_daily_value
            for i in range(42):
                perf_daily_value.append(np.dot(
                    number_stock, port_table.iloc[(index+42+i), :])-fund_borrowed+cash_optimal)

        return perf_daily_value

# plot portfolio daily value with benchmark


def portfolio_daily_value_2021_plot(perf_daily_value):
    plt.figure(figsize=(12, 9))
    benchmark_data = pd.read_csv('Daily_closing_prices_100.csv')['TLT']
    plt.plot(perf_daily_value, label='Suggested Portfolio')
    plt.plot(perf_daily_value[0]*(benchmark_data[439:691]/benchmark_data[439]).to_numpy(),
             label='20+ Year Treasury Bond')
    # x axis ticks
    #plt.xticks(list(dates[k] for k in list(np.linspace(0,504,10,dtype=int))),rotation=45)

    # naming the x axis
    plt.xlabel('Day')
    # naming the y axis
    plt.ylabel('Portfolio Daily Values per $10000($)')

    # giving a title to my graph
    plt.title('2021 Portfolio Daily Values')

    # function to show the plot
    plt.legend()
    plt.savefig('port_value.png')
    receive_picture('port_value.png')


# get the name of the current node
def get_node(response):
    return response['context']['skills']['main skill']['user_defined']['current_node']

# change the risk tolerance level in context variables


def change_risk_tol(response, result_risk_tol):
    risk_tol_list = ['aggressive', 'slightly aggressive',
                     'moderate', 'slightly conservative', 'conservative']
    risk_bin = [0.6, 1.2, 1.8, 2.4, 3.]
    response['context']['skills']['main skill']['user_defined']['risk_tol_level'] = risk_tol_list[np.digitize(
        result_risk_tol, risk_bin)]
    return response['context']


def change_portfolio(response, port_table):
    response['context']['skills']['main skill']['user_defined']['best_port'] = port_table.to_string(
        index=False)
    return response['context']

# recognize user's choice


def reg_choice(response):
    choice_list = ["choose_A", "choose_B", "choose_C", "choose_D"]
    if not response['output']['intents']:
        return False, -1
    elif response['output']['intents'][0]["intent"] in choice_list:
        user_choice = ord(response['output']['intents'][0]["intent"][-1])-65
        return True, user_choice
    else:
        return False, -1


def send_message_to_watson_assistant(ASSISTANT_ID, SESSION_ID, input_text, Context=None):
    response = assistant.message(assistant_id=ASSISTANT_ID, session_id=SESSION_ID, input={
                                 'message_type': 'text', 'text': input_text, 'options': {'return_context': True}}, context=Context).get_result()
    return response


# Initialize the assistant
assistant = AssistantV2(version='2021-11-27', authenticator=IAMAuthenticator(
    'dNelfqCRuncbzr66IeUU2w4pEl57aFGrT-zjzN2we2e9'))
assistant.set_service_url(
    'https://api.us-south.assistant.watson.cloud.ibm.com/instances/126cfafc-e801-4bac-be5c-69e43cdfc5e6')
ASSISTANT_ID = "a3ff3343-7336-4645-b37d-f193ffe0279b"
SESSION_ID = assistant.create_session(
    assistant_id=ASSISTANT_ID).get_result()["session_id"]

risk_assess_ans = []
binary_list = ['no', 'yes']
risk_tol_level = 1.5
ESG_option = 0
fund_avl = 1.
inv_horizon = 1.
sect_pref = 0
port_table = pd.DataFrame(
    np.array([['APPL', 'GOGL'], [10, 10]]).T, columns=['name', 'number'])


# ----------------------------------UI interface----------------------------------


user_name = 'Jeff'  # The name of user shown in the window
user_color = 'red'  # The inputs of user will be shown in specified user_color
robo_name = 'InvestBot'  # The name of robo advisor shown in the window
robo_color = 'blue'  # The inputs of robo advisor will be shown in specified robo_color
canvas_width = 858  # Unit: pixel
canvas_height = 585
text_area_width = 104  # Unit: height of a letter
text_area_height = 44  # Unit: width of a letter

# Declare global variable for the image. Otherwise, the image cannot be presented.
image_jpg = None


window = tk.Tk()  # Set up window object
window.title('Group 1 Robo advisor interface')
window.geometry('1560x715')  #


# Define a condition bar to show the condition of the chat
condition_text = tk.StringVar()
condition_text.set(robo_name+' is waiting for your reply.')
condition_label = tk.Label(master=window, height=2, width=60,
                           anchor=tk.CENTER, textvariable=condition_text)
condition_label.pack(side=tk.TOP, fill=tk.X)

# Define a frame to put the scrolledtext object and the canvas
text_canvas_frame = tk.Frame(master=window)
# Define a scrolledtext object to store the information from Robo and user
text = ScrolledText(master=text_canvas_frame,
                    height=text_area_height, width=text_area_width)
# The inputs of user/robo will be shown in specified user/robo color
text.tag_config(tagName='user_tag', font='Arial', foreground=user_color)
text.tag_config(tagName='robo_tag', font='Arial', foreground=robo_color)
text.pack(side=tk.LEFT)

# Define a canvas to present the picture
# background argument: set the color of background
canvas = tk.Canvas(master=text_canvas_frame,
                   width=canvas_width, height=canvas_height)
canvas.pack(side=tk.RIGHT)
text_canvas_frame.pack(fill=tk.X)

# Define a frame to put the button for send and the entry box
entry_button_frame = tk.Frame(master=window)
# Define the entry
# Set up tk.Entry object and put it on the frame
entry = tk.Entry(master=entry_button_frame, width=70)
entry.pack(side=tk.LEFT)
# Define the send button


def button1_command():
    # Change the state to waiting robot's reply.
    condition_text.set('Waiting for '+robo_name+'\'s reply... ')
    text_input = entry.get()  # Get the content of input
    # If index argument = 'insert'，insert at the cursor。chars is the string to insert
    text.insert(
        'end', datetime.datetime.now().strftime('%F %T')+' '+user_name, 'user_tag')  # Show the time of user input and user_name, using user style.
    text.insert(index='end', chars='\n')  # Switch to a new row
    text.insert('end', text_input, 'user_tag')  # Show the content of input
    text.insert(index='end', chars='\n')
    text.see(tk.END)  # Roll to the end automatically.
    entry.delete(first=0, last='end')  # Clear the entry box

    # Send the user's response to robo and react
    global old_response
    global response
    global risk_assess_ans
    global risk_tol_level
    global ESG_option
    global fund_avl
    global inv_horizon
    global sect_pref
    global port_table

    # These codes below are designed to know which node we are in.
    if get_node(old_response)[:2] == "RA":
        response = send_message_to_watson_assistant(
            ASSISTANT_ID, SESSION_ID, text_input, Context=None)
        risk_assess_ans.append(reg_choice(response)[1])
    elif get_node(old_response) == "Complete_risk_test":  # after risk asssessment
        # compute risk toleraence level
        risk_tol_level = np.mean(risk_assess_ans)
        response = send_message_to_watson_assistant(
            ASSISTANT_ID, SESSION_ID, text_input, Context=change_risk_tol(old_response, risk_tol_level))
    elif get_node(old_response) == "Invest_Horizon":
        # record amount of available fund
        fund_avl = float(old_response['output']['entities'][0]['value'])
        response = send_message_to_watson_assistant(
            ASSISTANT_ID, SESSION_ID, text_input, Context=None)
    elif get_node(old_response) == "Sector_Pref":
        # record investment horizon
        inv_horizon = float(old_response['output']['entities'][0]['value'])
        response = send_message_to_watson_assistant(
            ASSISTANT_ID, SESSION_ID, text_input, Context=None)
        if reg_choice(response)[0]:
            # record sector preference
            sect_pref = int(reg_choice(response)[1]+1)
    elif get_node(old_response)[:4] == "ESG_":
        ESG_option = binary_list.index(
            get_node(old_response)[4:])  # record ESG preference
        port_table = change_portfolio_v2(risk_tol_level, ESG_option, fund_avl, inv_horizon, sect_pref)[
            1].to_frame(name='number').reset_index(level=0).set_axis(['name', 'number'], axis=1, inplace=False)
        response = send_message_to_watson_assistant(
            ASSISTANT_ID, SESSION_ID, text_input, Context=change_portfolio(old_response, port_table))
    else:
        response = send_message_to_watson_assistant(
            ASSISTANT_ID, SESSION_ID, text_input, Context=None)

    old_response = response

    # Present the response to the GUI
    receive_text(response["output"]["generic"][0]["text"])
    if get_node(old_response) == "Provide_Port":
        # give a plot to show the past performance
        portfolio_daily_value_2021_plot(portfolio_daily_value_2021(
            inv_horizon, ESG_option, sect_pref, risk_tol_level))
        receive_text("Are you satisfied with our recommendation?")
    if get_node(old_response)[-3:] == 'end':
        receive_text(
            "This is the end of this conversation. Thank you for using. To start another conversation, say hello to me.")


button1 = tk.Button(master=entry_button_frame, text='Send',
                    command=button1_command)
button1.pack(side=tk.LEFT, padx=10)


# Define clear button

def button2_command():
    # index1 = 1.0 Means the first row, 0th letter. Delete whole text box content
    text.delete(index1='1.0', index2='end')
    canvas.delete('all')


button2 = tk.Button(master=entry_button_frame,
                    text='Clear', command=button2_command)
button2.pack(side=tk.LEFT, padx=10)
entry_button_frame.pack(side=tk.BOTTOM)

# Define receive text function


def receive_text(text_to_entry: str):
    text.insert(
        'end', datetime.datetime.now().strftime('%F %T')+' '+robo_name, 'robo_tag')  # Show the time of user input and user_name, using user style.
    text.insert(index='end', chars='\n')  # Switch to a new row
    text.insert('end', text_to_entry, 'robo_tag')  # Show the content of input
    text.insert(index='end', chars='\n')
    text.see(tk.END)  # Roll to the end automatically.
    # Change the state to waiting robot's reply.
    condition_text.set('Waiting for '+user_name+'\'s reply... ')
    global wait_user
    wait_user = True  # Change the flag

# Define receive picture function


def receive_picture(picture_path: str):
    global image_jpg  # Must use global variable.
    image_jpg = ImageTk.PhotoImage(Image.open(picture_path))
    canvas.delete('all')  # Clear the canvas
    canvas.create_image(int(canvas_width/2),
                        int(canvas_height/2), anchor='center', image=image_jpg)  # The picture is shown in the middle of the canvas.
    # Change the state to waiting robot's reply.
    condition_text.set('Waiting for '+user_name+'\'s reply... ')
    global wait_user
    wait_user = True  # Change the flag

# ---------------------------------------UI interface end---------------------------------------


# -----------------------------------------------------Main---------------------
receive_text("Welcome to InvestBot!")  # welcome to the InvestBot
receive_text("If you think it does not work as you expected, answer \'ok\'.")
text_input = "Hello"  # get the user's input
old_response = send_message_to_watson_assistant(
    ASSISTANT_ID, SESSION_ID, text_input, Context=None)  # get the response from watson assistant
receive_text(old_response["output"]["generic"][0]
             ["text"])
# The processes above are designed to initialize the global variables: response and old_response

window.mainloop()  # Keep refreshing the window
# delete session
response = assistant.delete_session(
    assistant_id='a3ff3343-7336-4645-b37d-f193ffe0279b', session_id=SESSION_ID).get_result()
