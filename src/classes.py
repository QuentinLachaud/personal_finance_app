import pandas as pd
import numpy as np
import datetime

class Asset:
            def __init__(self, name, value, contrib, growth_rate):
                self.name        = name
                self.value       = value
                self.contrib     = contrib
                self.growth_rate = growth_rate

            def future_value(self, nyears):
                self.nyears = nyears
                final_value = self.value * ((1 + self.growth_rate) ** self.nyears) + self.contrib * (((1 + self.growth_rate) ** self.nyears - 1) / self.growth_rate)

                return final_value
            
            def annual_projection(self, nyears, df=False):

                self.nyears = nyears
                returns     = []
                ret         = self.value

                for _ in range(self.nyears):
                    ret *= (1 + self.growth_rate / 100)
                    ret += self.contrib
                    returns.append(ret)
                if df:
                      returns = pd.DataFrame(returns, columns=[self.name], index=None)
                      return returns

                return returns
            
class Debt:
    def __init__(self, name, loan_amount, interest, term, monthly_overpayment=None, annual_payment=None, fixed=None):

        self.name                = name
        self.loan_amount         = loan_amount
        self.annual_payment      = annual_payment
        self.interest            = interest / 100
        self.monthly_interest    = self.interest / 12
        self.fixed               = fixed
        self.term                = term
        self.term_months         = term * 12
        self.annual_payment      = annual_payment
        self.monthly_overpayment = monthly_overpayment

        if self.monthly_overpayment != None:
             self.name += '_op'
             
    
    def calc_monthly_repayment(self):
        numerator           = self.loan_amount * self.monthly_interest * (1 + self.monthly_interest) ** self.term_months
        denominator         = ((1 + self.monthly_interest) ** self.term_months) - 1
        self.monthly_payment = numerator / denominator

        return self.monthly_payment
    
    def future_value(self, nyears):
        remaining_balance = self.loan_amount * ((1 + self.interest) ** nyears) - self.annual_payment * (((1 + self.interest) ** nyears - 1) / self.interest)
        return remaining_balance

    def annual_projection(self, nyears, df=False):
        if not self.annual_payment:
             self.annual_payment = self.calc_monthly_repayment() * 12
        remaining_balance = float(self.loan_amount)  # Convert to float
        balances = []

        if self.fixed:
             nyears = self.fixed

        for _ in range(nyears):
            remaining_balance *= (1 + self.interest)
            remaining_balance -= self.annual_payment
            if self.monthly_overpayment:
                remaining_balance -= self.monthly_overpayment * 12
            
            balances.append(float(remaining_balance))  # Convert to float

        if df:
            balances_df = pd.DataFrame(balances, columns=[self.name])
            return balances_df

        return balances


class streamlit_tab:
    def __init__(self, name, title=None, subtitle=None, description=None, body=None):
        self.name        = name
        self.title       = title
        self.subtitle    = subtitle
        self.description = description
        self.body        = body
    
    def render_self(self):
         for element in self.body:
             element
    

class retirement:
     def    __init__(self, age, retirement_age, net_worth, annual_expenses, annual_income, assets, debts):
         self.age             = age
         self.retirement_age  = retirement_age
         self.net_worth       = net_worth
         self.annual_expenses = annual_expenses
         self.annual_income   = annual_income
         self.assets          = assets
         self.debts           = debts


class MonteCarlo():
    """
    Parking this here to avoid losing it. Does not adapt to different rates year on year.
    Replicates the table generated on: https://www.thecalculatorsite.com/finance/calculators/compoundinterestcalculator.php
    """

    def __init__(self, principal=0, annual_contrib=20000, annual_rate=0.07, years=20, sims=2):

        self.principal      = principal
        self.annual_contrib = annual_contrib
        self.annual_rate    = annual_rate
        self.years          = years
        self.sims           = sims

        self.deposits            = np.full_like([i for i in range(years)], annual_contrib)
        self.cumulative_deposits = np.cumsum(self.deposits)
        self.account_balance     = [self.fv(t=i) for i in range(1, self.years + 1)]

        self.table = pd.DataFrame({'deposits': self.deposits,
                                   'cumulative_deposits': self.cumulative_deposits,
                                   'balance': self.account_balance,
                                   'accrued_interest': self.account_balance - self.cumulative_deposits
                                   })
        self.table['interest'] = self.table['accrued_interest'].diff()

        self.table            = self.table[['deposits', 'interest', 'cumulative_deposits', 'accrued_interest', 'balance']]
        self.table.index.name = 'Year'
        self.table            = self.table.astype(float)

        
    def fv(self, P=None, c=None, r=None, n=1, t=None):
        """Future value of a series of deposits and interest"""
        P = self.principal
        c = self.annual_contrib
        r = self.annual_rate

        if not t:
            t = self.years

        principal_component     = (P * (1 + r / n)**(n * t))
        contributions_component = (c * ((1 + r / n)**(n * t) - 1)) / (r / n)
        self.future_value = principal_component + contributions_component

        return self.future_value