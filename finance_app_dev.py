from gsheetsdb import connect
from matplotlib import pyplot as plt
from PIL import Image
from google.oauth2 import service_account
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import pandas as pd
import numpy as np
import time
import streamlit as st
import seaborn as sns
import datetime
from datetime import timedelta
import yfinance as yf
import time


import plotly.express as px
import plotly.graph_objects as go

from src.classes import Asset, Debt
from src.utils import project_returns, random_walk, make_net_worth_df, change_text_colour, generate_retirement_portfolio

from src.data import top_tickers
from src.classes import streamlit_tab

#################
# Page settings #
#################
st.set_page_config(layout="wide")

# page_style_toggle = st.toggle(label='Dark mode', value=False)
# if page_style_toggle == True:
#     st.toggle('Light Mode', value=True)
#     plt.style.use('default')
# else:
#     plt.style.use('dark_background')

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(['Investments', 'a', 'Net Worth', 'Mortgage', 'Stocks', 'debug', 'Early_retirement', 'Early Retirement_beta'])

#make a dict display_tab with bools
display_tab = {'tab1': True,
                'tab2': False,
                'tab3': True,
                'tab4': True,
                'tab5': True,
                'tab6': False,
                'tab7': True,
                'tab8': False
            }






if display_tab['tab1']:

    #########
    # TAB 1 #
    #########

    with tab1:


        st.title('Compound Interest Calculator')
        st.divider()
        input_col_1, graph_col = st.columns([1, 4]) 

        with input_col_1:
            start_date       = st.date_input('Start date')
            starting_capital = st.number_input('Starting capital', min_value=0, max_value=10000000, step=10000, value=10000)
            contribution     = st.number_input('Annual contributions', min_value=0, max_value=100000, step=1000, value=1200)
            time_period      = st.slider('Time period', 0, 100, 30, help='How many years in the future to project to.')
            return_rate      = st.slider('Return rate (%)', 0, 30, 6, help='Annual appreciation over inflation(%)')

            b = st.checkbox(label='contributions', value=True)
            c = st.checkbox(label='accumulated interest', value=True)
                    
        # Calcs
        df       = project_returns(capital=starting_capital, annual_rate=return_rate / 100, years=time_period, annual_contribution=contribution)
        df.index = [i.year for i in pd.date_range(str(start_date.year), freq= 'Y', periods=time_period)]
        
        df['contrib']                = contribution
        df['contributions']          = df['contrib'].cumsum()
        df['accumulated interest']   = (df['total_value'].diff() - contribution).fillna(0).cumsum()
        df['annual interest income'] = df['accumulated interest'].diff().fillna(0)


        with graph_col:
            

            # variables for text display
            years              = time_period
            total_value        = df['total_value'].tail(1).values[0]
            interest_income    = df['accumulated interest'].tail(1).values[0]
            pct_interest_total = np.round(interest_income / total_value * 100, 2)

            #Visuals and text
            st.write(f'Projection of £ {starting_capital:,.0f} invested capital over {time_period} years.')
            if contribution == 0:
                st.write(f'This includes a gross return rate of {return_rate} %.')
            else:
                st.write(f'This includes a gross return rate of {return_rate} %, with annual contributions of £ {int(contribution):,}. (£ {int(contribution/12)} / month.)')
            st.write(f"The projected value after {years} years is £ {int(total_value):,.0f}. £ {int(interest_income):,.0f} earned from interest alone, or {pct_interest_total} % of all gains.")
        
            # Create charts based on selected checkboxes
            selected_columns = []

            
            if b:
                selected_columns.append('contributions')
            if c:
                selected_columns.append('accumulated interest')
            
            
            if selected_columns:
                st.bar_chart(df[selected_columns])
            
            df.index = df.index.astype(str)
            with st.expander('See table'):
                st.download_button(':floppy_disk:  Download table', str(df), file_name='compound_interest_projection.csv')
                st.table(df)

        
        
        space_1, table_space, space_2 = st.columns([4, 8, 4])
  

        # Tab1 END -------------------------------------------------

if display_tab['tab2']:

    #########
    # TAB 2 #
    #########

    with tab2:

        st.title('Randomness in action')
        st.divider()
        st.markdown('Select a place to start, and a number of steps to walk.')
        st.markdown('A random sequence will be generated, where each steps either adds or subtracts 1 from the latest value.')
        st.markdown(' ')

    
        
        col1, col2= st.columns([1,2])
        with col1:
            start_point = st.number_input('Where the walk begins', 0, 1000)
        with col2:
            end_point = st.slider('How far is the walk?', 100, 10000, step=100)

    
        walk = random_walk(start_point=start_point, walk_to=end_point, step=1)

        col1, col2 = st.columns([3, 2])
        with col1:
            st.line_chart(walk)
            
            
        with col2:
            fig = plt.figure()
            sns.histplot(walk, color='orange', edgecolor='white')
            st.pyplot(fig)

        st.title('Multiple simulations')
        sim_count = st.slider('How many simulations would you like?', 1, 100, 10)

        sims = [random_walk(start_point=0, walk_to=1000, step=1, df=False) for i in range(sim_count)]
        sims_df = pd.DataFrame(columns=[f'sim_{i}' for i in range(len(sims))], data=np.array(sims).T)
        

        st.line_chart(sims_df, use_container_width=True)




    # TAB2 END -------------------------------------------------
if display_tab['tab3']:
        
    #########
    # TAB 3 #
    #########

    with tab3:

        st.title('Net Worth Calculator')
        st.write(':grey[_Project your net worth using your current assets and liabilities._]', help='Historical average % returns are used as baselines. Adjust as needed.')
        st.divider()

        col1, col2, col3 = st.columns([2, 2, 10])
        with col1:
            dob = st.number_input('What year were you born?', min_value=1900, value=1991, step=1 ,key='age')
            
        with col3:
            net_worth_slider = st.slider(label='Years into the future', min_value=1, max_value=80, step=1, key='project_future1')    
                
        col1, col2, col3, col4 = st.columns([1.5, 1, 1.5, 10])  # input, return, contrib, graphs
                
        with col1:
                    
            # Assets value
            cash_input      = st.number_input('Cash savings', value=1000, min_value=0, step=1000, key='cash')
            pension_input   = st.number_input('Pension value today', value=1000, min_value=0, step=1000, key='pension')
            stocks_input    = st.number_input('Stocks invested', value=1000, min_value=0, step=1000, key='stocks')
            property_input  = st.number_input('Property value', value=1000, min_value=0, step=1000, key='property')
            crypto_input    = st.number_input('Crypto value', value=1000, min_value=0, step=1000, key='crypto')
            debt_input      = st.number_input('Outstanding debt', value=-1000, max_value=0, step=-1000, key='debt', help='Enter a negative value')

        with col2:

            # Assets expected annual return
            empty = st.empty()
            cash_ret      = st.number_input('% return', value=-3.0, min_value=-10.0, step=.5, key=1)
            pension_ret   = st.number_input('% return', value=5.0, min_value=-10.0, step=.5, key=2)
            stocks_ret    = st.number_input('% return', value=8.0, min_value=-10.0, step=.5, key=3)
            property_ret  = st.number_input('% return', value=3.5, min_value=-10.0, step=.5, key=4)
            crypto_ret    = st.number_input('% return', value=3.0, min_value=-10.0, step=.5, key=5)
            debt_ret      = st.number_input('% return', min_value=4.0, step=.5, key=6, help='Your mortgage rate')

        with col3:

            # Assets annual contribution
            cash_contrib      = st.number_input('Annual contrib.', value=0, min_value=0, step=100, key='cash2')
            pension_contrib   = st.number_input('Annual contrib.', value=0, min_value=0, step=1000, key='pension2')
            stocks_contrib    = st.number_input('Annual contrib.', value=0, min_value=0, step=1000, key='stocks2')
            property_contrib  = st.number_input('Annual contrib.', value=0, min_value=0, step=1000, key='property2')
            crypto_contrib    = st.number_input('Annual contrib.', value=0, min_value=0, step=1000, key='crypto2')
            debt_contrib      = st.number_input('Annual contrib.', min_value=0, step=100, key='debt2')

        with col4:

            # Graphs and text
            current_year     = datetime.datetime.now().year
            age              = current_year - dob
            projection_year  = net_worth_slider + current_year  # how far the projection goes to
            projection_age   = net_worth_slider + age  # how old you will be
            color            = change_text_colour(projection_age)

            st.write(f"In {projection_year} you will be :{color}[{projection_age} years old.]")
            
            if net_worth_slider + age >= 100:
                st.markdown(':grey[Remember you\'re probably] :red[dead] :grey[at this point...]' +  ':skull_and_crossbones:')        

            # Instantiate Assets
            cash     = Asset('cash', cash_input, cash_contrib, cash_ret)
            pension  = Asset('pension', pension_input, pension_contrib, pension_ret)
            stocks   = Asset('stocks', stocks_input, stocks_contrib, stocks_ret)
            property = Asset('property', property_input, property_contrib, property_ret)
            crypto   = Asset('crypto', crypto_input, crypto_contrib, crypto_ret)
            debt     = Asset('debt', debt_input, debt_contrib, debt_ret)



            assets = [cash, pension, stocks, property, crypto, debt]

            net_worth_df       = make_net_worth_df(assets=assets, project_years=net_worth_slider)
            net_worth_df.index = [str(i.year) for i in pd.date_range(str(current_year + 1), periods=len(net_worth_df), freq='YS')]
            future_net_worth   = net_worth_df.iloc[-1, :].sum()

    # Tab 3 Outputs ------------------------------------------------------------------------------------------------------------

            with col1:
        
            # Net worth text output
                net_worth = cash_input + pension_input + stocks_input + property_input + crypto_input + debt_input

            graph_col, graph_col2 = st.columns([1, 1])

            with graph_col:
                st.write("Absolute value over time")
                st.bar_chart(net_worth_df)

            with graph_col2:
                st.write("Relative allocation over time")
                rel_alloc_df = net_worth_df.copy(deep=True)
                rel_alloc_df['debt'] = 0
                st.bar_chart(rel_alloc_df.apply(lambda x: x / np.sum(x) * 100, axis=1))
            
        col1, col2 = st.columns([4, 8])
        with col1:
            st.divider()
            st.write(f'Your current net worth is £ **{net_worth:,.0f}**', )
            st.write(f'Your net worth in {net_worth_slider} years will be £ {future_net_worth:,.0f}.')

        with col2:
            with st.expander('-- See table -- '):
                st.download_button(':floppy_disk:  Download Net Worth Table', str(net_worth_df), file_name='net_worth_{net_worth_slider}_year_projection.csv')
                st.table(net_worth_df)

    # TAB3 END -------------------------------------------------

if display_tab['tab4']:

    
    #########
    # TAB 4 #
    #########
    
    with tab4:

        
        st.title('Mortgage payments calculator')
        st.write(':grey[_Enter your loan size and interest. See your payment plan and how much interest/time can be saved by overpaying._]')

        col1, col2, col3 = st.columns([5, 10, 1.5])

        with col1:
            loan_amount = st.number_input('Mortgage size', value=201000)
            interest    = st.number_input('Annual interest (e.g. 3.5%)', value=4.98, step=0.01)
            term        = st.number_input('Mortgage duration (years)', value=20)
            st.divider()
            monthly_overpayment = st.slider('Overpayment (£ monthly)', value=0, min_value=0, max_value=500, step=25, help='It is worth noting you will _almost always_ be better off investing in a Global all-cap fund')
            st.divider()

            
            mortgage_house     = Debt('house mortgage', loan_amount=loan_amount, term=term, interest=interest, monthly_overpayment=None)
            mortgage_house_op  = Debt('house mortgage', loan_amount=loan_amount, term=term, interest=interest, monthly_overpayment=monthly_overpayment)

            repayment        = mortgage_house.calc_monthly_repayment() + monthly_overpayment
            repayment_widget = st.metric('Monhtly Repayment:', value=f'£ {np.round(repayment, 0)}', delta=f'+ £ {monthly_overpayment} overpayment')

            mortgage_house_df    = mortgage_house.annual_projection(term, df=True)
            mortgage_house_df_op = mortgage_house_op.annual_projection(term, df=True)

            if monthly_overpayment > 0:

                output_df                   = pd.concat([mortgage_house_df, mortgage_house_df_op], axis=1)
                output_df.columns           = ['No overpayment', f'£{monthly_overpayment} extra per month']
                output_df['interest saved'] = np.abs(output_df.iloc[:, 0] - output_df.iloc[:, 1] - monthly_overpayment * 12)
                
                output_df = output_df[output_df[f'£{monthly_overpayment} extra per month'] > 0]
               
                output_df = output_df.apply(lambda x: np.round(x, 0))

                
                

                years       = len(output_df)#[output_df[f'£{monthly_overpayment} extra per month'] != 0])
                years_saved = term - years

                total_paid = int(repayment * 12 * years) + monthly_overpayment * 12 * years

                st.markdown(f"- Overpaying £ {monthly_overpayment} per month could save £ {output_df['interest saved'].iloc[-1]:,.0f} over the full term.")
                st.markdown(f"- You will pay £ {total_paid:,.0f} over {years} years of your mortgage, saving {years_saved} {'year' if years_saved < 2 else 'years'}.")
                st.markdown(f'- (£{total_paid / loan_amount:,.2f} per £1 borrowed.)')

                
            else:
                total_paid = int(repayment * 12 * term)
                output_df  = mortgage_house_df

                st.write(f"You will pay £ {total_paid:,.0f} over the full {term} years of your mortgage.")
                st.write(f'(£{total_paid / loan_amount:,.2f} per £1 borrowed.)')

                
        with col3:
            chart_type = st.radio('Chart type', ['line', 'bar'], index=0)

        with col2:
            st.markdown('### Loan repayment projection')

            if chart_type == 'line':
                st.line_chart(output_df)

            elif chart_type == 'bar':
                st.bar_chart(output_df)

            with st.expander('See breakdown'):
                st.download_button(':floppy_disk:  Download table', str(output_df), file_name='mortgage_repayment_projection.csv')
                st.table(output_df)

    # TAB4 END -------------------------------------------------

if display_tab['tab5']:
    
        #########
        # TAB 5 #
        #########

    with tab5:
        st.title('Display historical stock data of top tickers')
        st.write(':grey[_Select a date range and a list of stocks to display._]')
        st.divider()

        today = datetime.datetime.now().strftime(format='%Y-%m-%d')

        # Retrieve historical data
        sp500               = yf.download('^GSPC', start='1985-01-01', end='2025-04-01')
        sp500.columns       = [i.lower() for i in sp500.columns]
        sp500['close_diff'] = sp500['close'].pct_change()
        
        col1, col2 = st.columns([1, 2])

        with col1:

            # Widgets
            date_start        = st.date_input(label='start', value=datetime.datetime(2000,1,1), max_value=datetime.datetime.now() - datetime.timedelta(days=1))
            date_end          = st.date_input(label='end', max_value=datetime.datetime.now())
            select_stocks     = st.multiselect('Choose your poison', options=list(top_tickers.keys()))
            # get_stocks_button = st.button('Get sToNKs now')
            get_all_stocks = None
            

            date_start_str = date_start.strftime('%Y_%m_%d').replace('-', '_')
            date_end_str   = date_end.strftime('%Y_%m_%d').replace('-', '_')

            stocks_df = yf.download(list(top_tickers.keys()), start=date_start, end=date_end)['Close'].resample('D').mean().ffill()
            
            if not select_stocks:
                if get_all_stocks != None:
                    st.error(':small_red_triangle: :gray[_select some stocks_] :small_red_triangle:')
            get_all_stocks    = st.checkbox('Get ALL the StoNKs')

            if select_stocks:
                stocks_df = stocks_df[select_stocks]

            if get_all_stocks:
                with col2:
                    with st.spinner('Fetching all the sToNkS...'):
                        time.sleep(1)
                stocks_df = stocks_df[top_tickers.keys()]
                # stocks_df = stocks_df[top_tickers.keys()]
            
        
        with col2:
            from src.data import intervals
            smoothing_interval = st.select_slider('interval', options=intervals.keys())
                                        
            resampling_period = intervals[smoothing_interval]

            if (select_stocks or get_all_stocks) and smoothing_interval:
                smoothed_df = stocks_df.resample(resampling_period).mean()
                st.line_chart(smoothed_df)

            if select_stocks or get_all_stocks:
                with st.expander('-- view full table --'):
                    st.download_button('Download raw data', data=str(stocks_df), file_name=f"{'_'.join(select_stocks)}_{date_start_str}_to_{date_end_str}.csv")
                    st.table(stocks_df)

    # TAB5 END -------------------------------------------------
if not display_tab['tab6']:
    with tab6:
        st.write('Debug tab is not active right now.')

if display_tab['tab6']:

    #########
    # TAB 6 #
    #########

    with tab6:

        debug1, debug2 = st.columns(([1, 2]))
        with debug1:
            fucksliders = st.slider('fuck sliders how much?', min_value=2, max_value=90, value=1)
            raw_df = yf.download(['AAPL', 'MSFT'], start='2018-01-01', end='2021-01-01')["Close"]
            resampled_df = raw_df.resample(f'{fucksliders}D').mean()
            # st.table(resampled_df)
        
            st.line_chart(resampled_df)
    
    # TAB 6 END ---------------------------------------------------------------

if display_tab['tab7']:

    #########
    # TAB 7 #
    #########

    with tab7:
        import streamlit_vertical_slider as svs

        title_col1, title_col2 = st.columns([1, 1])
        with title_col1:
            st.title('Retirement calculator original')
            st.write(':grey[_Project retirement viability from investments using Monte Carlo simulation_]')
        st.divider()

        col1, col2 = st.columns([3, 10])

        with col1:
            inflation_toggle = st.toggle('Adjust for inflation?', value=False)
            if inflation_toggle:
                st.write(':yellow[_Showing real returns_]', help='Projection will show value with today\s buying power')
            else:
                st.write(':red[Results *not* adjusted for inflation]')
        # with col2:
        #     # st.link_button('Work out expendable income', url='https://docs.google.com/spreadsheets/d/1q5mhV50sPa6dO5N37aEoIM5np94mJ3Uo56WKfbhEDBY/edit#gid=1215540826')
        #     st.empty()

        with col1:

            investment    = st.number_input('Initial investment', value=0, min_value=0, max_value=10000000, step=1000)
            start_year    = st.date_input('start date', value=pd.datetime(2024,1,1))
            simulations   = st.number_input('Simulations', value=1000, min_value=1, max_value=100000, step=100)
            years         = st.number_input('Years to project', value=20, min_value=1, max_value=100, step=1)

            periods       = st.number_input('Periods per year', value=1, min_value=1, max_value=365, step=1)
            inflation     = 3.8 / periods

            # Ins and Outs
            contributions = st.number_input('Annual contributions', value=10000, min_value=-500000, max_value=500000, step=1000)
        
            st.divider()

            avg_mkt_ret  = st.number_input('Annual market return %', value=10.0, min_value=-10.0, max_value=30.0, step=1.0) / periods
            avg_mkt_vol  = st.number_input('Annual market volatility %', value=15.0, min_value=0.0, max_value=50.0, step=5.0) / np.sqrt(periods)

            # CALCS

            # Contributions
            contributions    = np.full(years * periods, contributions / periods)
            # contributions[0] = investment                                           # Set the first value to the initial investment
            contributions    = contributions.cumsum().reshape(-1, 1)            # List of cumulative annual_change as a column vector

            # Inflation
            if inflation_toggle:
                avg_mkt_ret -= inflation

            returns_mult = np.random.normal(loc=avg_mkt_ret / 100, scale=avg_mkt_vol / 100, size=(years * periods, simulations)) + 1
            returns_mult = np.cumprod(returns_mult, axis=0)

            # Portfolio value
            if investment > 0:
                portfolio_value_df = pd.DataFrame(investment * returns_mult + contributions, columns=[f'sim_{i}' for i in range(simulations)]) 
            else:
                portfolio_value_df = pd.DataFrame(returns_mult * contributions, columns=[f'sim_{i}' for i in range(simulations)])

            # Store last row to show outcomes for each simulation
            outcomes    = portfolio_value_df.iloc[-1, :].sort_values(ascending=True)
            success_pct = np.mean(outcomes > 0) * 100

            # Calculate the distribution metrics per period
            spreads_df         = pd.DataFrame(list(portfolio_value_df.apply(lambda x: np.percentile(x, [90, 75, 50, 25, 10]), axis=1)))
            spreads_df.columns = ['90%ile', '75%ile', 'median', '25%ile', '10%ile']
            
            df_with_spreads       = pd.concat([portfolio_value_df, spreads_df], axis=1)
            df_with_spreads       = pd.concat([df_with_spreads, pd.DataFrame({'Contributions': contributions.reshape(-1)})], axis=1)
            df_with_spreads.index = pd.date_range(start=start_year, periods=years * periods, freq=f'{int(365 / periods)}D')    
                        
            with col2:
                # Describe rate of success
                st.metric('Success rate', value=f'{success_pct:.1f} %', delta=None, help=f'Percentage of simulations that ended with a positive portfolio value after {years} years')

                # Plot the figure
                sims_tick = st.checkbox('Show all simulations', value=False, help='Showing all traces slows things down a bit.')
                if sims_tick: 

                    fig = px.line(df_with_spreads, 
                                title='Simulation of investment growth',
                                color_discrete_sequence=['blue'] * simulations + ['orange', 'yellow', 'red', 'yellow', 'orange', 'lime'])

                    # Adjust the traces
                    for trace in fig.data:

                        if trace.name in ['10%ile', '25%ile', '75%ile', '90%ile']:
                            trace.opacity = .7
                            trace.showlegend = True
                            trace.line.dash = 'dot'

                        elif trace.name == 'median':
                            trace.opacity = 1
                            trace.showlegend = True

                        elif trace.name == 'Contributions':
                            trace.opacity = 1
                            trace.showlegend = True

                        
                        else: 
                            trace.opacity = 0.4  # Set opacity to 0.4 for other traces
                            trace.showlegend = False  # Hide legend for other traces
                    
                    # Step line for annual_change
                    fig.update_traces(selector=dict(name='Contributions'), line_shape='hv')

                    fig.update_layout(yaxis=dict(range=[None, df_with_spreads['90%ile'].max() * 1.2]))


                    # Set the figure dimensions
                    fig.update_layout(
                        width=1000,  
                        height=600,
                    )

                    # Plot the figure using Streamlit
                    st.plotly_chart(fig)

                else:
                    fig = px.line(df_with_spreads[['90%ile', '75%ile', 'median', '25%ile', '10%ile', 'Contributions']], 
                                title='Simulation of investment growth',
                                color_discrete_sequence=['yellow', 'orange', 'red', 'orange', 'yellow', 'lime'])

                    # Adjust the traces
                    for trace in fig.data:

                        if trace.name in ['10%ile', '25%ile', '75%ile', '90%ile']:
                            trace.opacity = .7
                            trace.showlegend = True
                            trace.line.dash = 'dot'

                        elif trace.name == 'median':
                            trace.opacity = 1
                            trace.showlegend = True

                        elif trace.name == 'Contributions':
                            trace.opacity = 1
                            trace.showlegend = True
                    
                    # Step line for annual_change
                    fig.update_traces(selector=dict(name='Contributions'), line_shape='hv')

                    fig.update_layout(yaxis=dict(range=[None, df_with_spreads['90%ile'].max() * 1.2]))


                    # Set the figure dimensions
                    fig.update_layout(
                        width=1000,  
                        height=600,
                    )

                    # Plot the figure using Streamlit
                    st.plotly_chart(fig)

        with col1:
            with st.expander('-- See outcomes table --'):
                st.download_button(':floppy_disk:  Download table', str(spreads_df), file_name=f'{years}_year_{avg_mkt_ret}_retirement_outcomes.csv')
                spreads_df.index.name = 'Year'
                st.table(df_with_spreads[['Contributions', 'median', '10%ile', '25%ile', '75%ile', '90%ile']])

    # Title level metrics here as calcs needed to be done after the df was created
    with title_col2:
        title_subcols = st.columns([1, 1, 1, 1])

        with title_subcols[0]:
            st.metric('Time horizon', value=f'{years} years')

        with title_subcols[1]:
            if inflation_toggle:
                real_return = st.metric('Real return', value = f'{avg_mkt_ret * periods:.1f} %', delta = f'{- inflation * periods} %')
            else:
                real_return = st.metric('Unadjusted returns', value = f'{(avg_mkt_ret * periods)} %', delta = f'{inflation * periods} %')

        with title_subcols[2]:
            median_outcome = int(df_with_spreads['median'].iloc[-1])
            gains   = median_outcome - investment
            st.metric('Median Outcome', value=f"£ {median_outcome:,.0f}", delta=f"£ {gains:,.0f}", delta_color='normal')
        
        with title_subcols[3]:
            st.metric('Simulations', value=f'{simulations}')


    


# if display_tab[tab8]:

#         #########
#         # TAB 8 #
#         #########
#     with tab8:
#         import streamlit_vertical_slider as svs

#         title_col1, title_col2 = st.columns([1, 1])
#         with title_col1:
#             st.title('Retirement calculator')
#             st.write(':grey[_Project retirement viability from investments using Monte Carlo simulation_]')
#         st.divider()

#         col1, col2 = st.columns([3, 10])

#         with col1:
#             inflation_toggle = st.toggle('Adjust for inflation?', value=False, key='inflation_toggle2')
#             if inflation_toggle:
#                 st.write(':yellow[_Showing real returns_]', help='Projection will show value with today\s buying power')
#             else:
#                 st.write(':red[Results *not* adjusted for inflation]')
#         # with col2:
#         #     # st.link_button('Work out expendable income', url='https://docs.google.com/spreadsheets/d/1q5mhV50sPa6dO5N37aEoIM5np94mJ3Uo56WKfbhEDBY/edit#gid=1215540826')
#         #     st.empty()

#         with col1:

#             investment    = st.number_input('Initial investment', value=76000, min_value=0, max_value=10000000, step=1000)
#             start_year    = st.date_input('start date', value=pd.datetime(2024,1,1))
#             simulations   = st.number_input('Simulations', value=100, min_value=10, max_value=100000, step=100)

#             contribution_years = st.number_input('Years to contribute', value=15, min_value=0, max_value=100, step=1)
#             stagnate_years     = st.number_input('Years to stagnate', value=5, min_value=0, max_value=100, step=1)
#             drawdown_years     = st.number_input('Years to drawdown', value=40, min_value=0, max_value=100, step=1)

#             min_horizon = contribution_years + stagnate_years + drawdown_years

#             time_horizon  = st.number_input('Years to project', value=min_horizon, min_value=min_horizon, max_value=150, step=1)
#             periods       = st.number_input('Periods per year', value=1, min_value=1, max_value=365, step=1)
            
#             inflation     = 3.8 / periods

#             # Ins and Outs
#             contributions = st.number_input('Annual contributions', value=20000, min_value=-100000, max_value=100000, step=1000)
#             drawdowns     = st.number_input('Annual drawdowns', value=20000, min_value=0, max_value=100000, step=1000)
#             contrib_per_period   = contributions / periods
#             drawdowns_per_period = - drawdowns / periods

#             stagnations = [0] * stagnate_years * periods

#             st.divider()

#             avg_mkt_ret  = st.number_input('Annual market return %', value=10.0, min_value=-10.0, max_value=30.0, step=1.0) / periods
#             avg_mkt_vol  = st.number_input('Annual market volatility %', value=15.0, min_value=0.0, max_value=50.0, step=5.0) / np.sqrt(periods)

#             # CALCS

#             # Contributions
#             annual_change    = np.array([contrib_per_period] * periods * contribution_years + [stagnate_years] * periods * stagnate_years + [drawdowns_per_period] * periods * drawdown_years )  # Creating a list of annual_change per period
#             annual_change[0] = investment                                           # Set the first value to the initial investment
#             annual_change    = np.cumsum(annual_change).reshape(-1, 1)              # List of cumulative annual_change as a column vector

#             # Inflation
#             if inflation_toggle:
#                 avg_mkt_ret -= inflation

#             returns_mult = np.random.normal(loc=avg_mkt_ret / 100, scale=avg_mkt_vol / 100, size=(time_horizon * periods, simulations)) + 1
#             returns_mult = np.cumprod(returns_mult, axis=0)

#             # Portfolio value
#             portfolio_value_df = pd.DataFrame(annual_change * returns_mult, columns=[f'sim_{i}' for i in range(simulations)]) 

#             # Store last row to show outcomes for each simulation
#             outcomes = portfolio_value_df.iloc[-1, :].sort_values(ascending=False)

#             # Calculate the distribution metrics per period
#             spreads_df = pd.DataFrame(list(portfolio_value_df.apply(lambda x: np.percentile(x, [90, 75, 50, 25, 10]), axis=1)))
#             spreads_df.columns = ['90%ile', '75%ile', 'median', '25%ile', '10%ile']
            
#             df_with_spreads = pd.concat([portfolio_value_df, spreads_df], axis=1)
#             df_with_spreads = pd.concat([df_with_spreads, pd.DataFrame({'annual_change': annual_change.reshape(-1)})], axis=1)
#             df_with_spreads.index = pd.date_range(start=start_year, periods=time_horizon * periods, freq=f'{int(365 / periods)}D')
            
#             with col2:
#                 # Plot the figure
#                 fig = px.line(df_with_spreads[['10%ile', '25%ile', 'median', '75%ile', '90%ile', 'annual_change']], title='Simulation of investment growth', color_discrete_sequence=['orange', 'yellow', 'orange', 'orange', 'yellow', 'lime'])

#                 # Adjust the traces
#                 for trace in fig.data:

#                     if trace.name in ['10%ile', '25%ile', '75%ile', '90%ile']:
#                         trace.opacity = .7
#                         trace.showlegend = True
#                         trace.line.dash = 'dash'

#                     if trace.name in ['median', '75%ile']:
#                             trace.opacity = .5  # Reduce opacity for the area
#                             trace.fill = 'tonexty'  # Fill area to the next line (median)
#                             trace.showlegend = True

#                     # elif trace.name in ['25%ile', '75%ile']:
#                     #     trace.opacity = .5  # Reduce opacity for the area
#                     #     trace.fill = 'tonexty'  # Fill area to the zero line
#                     #     trace.showlegend = True

#                     elif trace.name == 'median':
#                         trace.color = 'red'
#                         trace.opacity = 1
#                         trace.showlegend = True

#                     elif trace.name == 'annual_change':
#                         trace.opacity = 1
#                         trace.showlegend = True
                    
#                     else:  
#                         trace.opacity = 0.4  # Set opacity to 0.4 for other traces
#                         trace.showlegend = False  # Hide legend for other traces
                
#                 # Step line for annual_change
#                 fig.update_traces(selector=dict(name='annual_change'), line_shape='hv')

#                 fig.update_layout(yaxis=dict(range=[0, df_with_spreads['90%ile'].max() * 1.2]))


#                 # Set the figure dimensions
#                 fig.update_layout(
#                     width=1000,  
#                     height=900,
#                 )

#                 # Plot the figure using Streamlit
#                 st.plotly_chart(fig)
            
#                 sns.histplot(outcomes, color='orange', edgecolor='white')
#                 plt.show()
        
#         with col1:
#             with st.expander('-- See outcomes table --'):
#                 st.download_button(':floppy_disk:  Download table', str(spreads_df), file_name=f'{time_horizon}_year_{avg_mkt_ret}_retirement_outcomes.csv')
#                 spreads_df.index.name = 'Year'
#                 st.table(df_with_spreads[['annual_change', 'median', '10%ile', '25%ile', '75%ile', '90%ile']])

#     # Title level metrics here as calcs needed to be done after the df was created
#     with title_col2:
#         title_subcols = st.columns([1, 1, 1])

#         with title_subcols[0]:
#             st.metric('Time horizon', value=f'{time_horizon} years')

#         with title_subcols[1]:
#             if inflation_toggle:
#                 real_return = st.metric('Real return', value = f'{avg_mkt_ret * periods:.1f} %', delta = f'{- inflation * periods} %')
#             else:
#                 real_return = st.metric('Unadjusted returns', value = f'{(avg_mkt_ret * periods)} %', delta = f'{inflation * periods} %')

#         with title_subcols[2]:
#             median_outcome = int(df_with_spreads['median'].iloc[-1])
#             gains   = median_outcome - investment
#             st.metric('Median Outcome', value=f"£ {median_outcome:,.0f}", delta=f"£ {gains:,.0f}", delta_color='normal')
