import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys

# Debug logging control - set to True to enable detailed logging
DEBUG_LOGGING = False  # Default to False for production use

# Setup logger for retirement calculator debugging (only if enabled)
if DEBUG_LOGGING:
    import os
    log_file_path = os.path.join(os.getcwd(), 'retirement_debug.log')

    # Clear any existing handlers to avoid conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),  # Console output
            logging.FileHandler(log_file_path, mode='a')  # Append mode to preserve logs
        ],
        force=True  # Force reconfiguration
    )
    logger = logging.getLogger(__name__)
    logger.info("=== LOGGING INITIALIZED ===")
    logger.info(f"Log file location: {log_file_path}")
else:
    # Create no-op logger when debugging is disabled
    logger = logging.getLogger(__name__)
    logger.disabled = True

"""
Multi-Currency Retirement Needs Calculator (Streamlit)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Estimate the nest‑egg required to retire at a selected age, using empirically observed
spending‑decline patterns and country-specific Social Security timing and benefits.

Supported Countries/Currencies:
- USA (USD): Social Security
- Europe (EUR): Average European pension systems  
- UK (GBP): State Pension
- Israel (NIS): Old Age Pension (Bituach Leumi)

How to use
==========
1. Select your country/currency
2. Adjust the inputs in the sidebar; the chart & figures refresh automatically
3. The app will use country-specific default values and social security calculations

Assumptions
-----------
* Spending declines 1%/yr from ages 62‑64 ("go‑go" -> "slow‑go" years).
* Real spending declines thereafter:
  • Couples: ‑2.4%/yr • Singles: ‑1.7%/yr  (RAND 2023, CRR 2021).
* Country-specific Social Security benefit multipliers and retirement ages
* Retirement portfolio ROI: 7% real during retirement
* Lifespan projects through age 95.
* Spousal Social Security benefits vary by country
"""

MAX_AGE = 95
EARLY_DECLINE = 0.01  # 1% p.a. ages 62‑64
COUPLE_DECLINE = 0.024  # 2.4% p.a. couples ≥65
SINGLE_DECLINE = 0.017  # 1.7% p.a. singles ≥65
POST_RET_ROI = 0.07  # real return assumption in retirement

# Country configurations
COUNTRY_CONFIG = {
    "USA": {
        "currency": "USD",
        "symbol": "$",
        "ss_full_age": 67,
        "ss_early_age": 62,
        "ss_late_age": 70,
        "ss_early_reduction": 0.06,  # 6% per year before FRA
        "ss_late_bonus": 0.08,  # 8% per year after FRA
        "default_monthly_spend": 5000,
        "default_ss_benefit": 2000,  # Monthly benefit at FRA
        "spousal_factor": 0.5,  # 50% spousal benefit
        "ss_name": "Social Security"
    },
    "Europe": {
        "currency": "EUR",
        "symbol": "€",
        "ss_full_age": 65,
        "ss_early_age": 60,
        "ss_late_age": 70,
        "ss_early_reduction": 0.04,  # 4% per year before FRA (varies by country)
        "ss_late_bonus": 0.05,  # 5% per year after FRA (varies by country)
        "default_monthly_spend": 4200,  # ~€1,345 average EU pension
        "default_ss_benefit": 1345,  # Average EU pension per month
        "spousal_factor": 0.6,  # 60% spousal benefit (varies by country)
        "ss_name": "State Pension"
    },
    "UK": {
        "currency": "GBP",
        "symbol": "£",
        "ss_full_age": 66,
        "ss_early_age": 66,  # No early claiming in UK
        "ss_late_age": 70,
        "ss_early_reduction": 0.0,  # No early claiming
        "ss_late_bonus": 0.05,  # ~5% per year deferral bonus
        "default_monthly_spend": 3500,  # ~£962 full state pension
        "default_ss_benefit": 962,  # £230.25/week = ~£962/month
        "spousal_factor": 1.0,  # Individual entitlement in UK
        "ss_name": "State Pension"
    },
    "Israel": {
        "currency": "NIS",
        "symbol": "₪",
        "ss_full_age": 67,
        "ss_early_age": 62,
        "ss_late_age": 70,
        "ss_early_reduction": 0.06,  # 6% per year before FRA
        "ss_late_bonus": 0.05,  # 5% per year after FRA
        "default_monthly_spend": 8000,  # ~₪1,795 individual pension
        "default_ss_benefit": 1795,  # NIS 1,795 individual pension (2025)
        "spousal_factor": 0.5,  # ₪902 spousal increment
        "ss_name": "Old Age Pension"
    }
}


def ss_multiplier(claim_age: int, country_config: dict) -> float:
    """Calculate Social Security benefit factor vs. claiming age for specific country."""
    fra = country_config["ss_full_age"]
    early_age = country_config["ss_early_age"]
    late_age = country_config["ss_late_age"]
    early_reduction = country_config["ss_early_reduction"]
    late_bonus = country_config["ss_late_bonus"]
    
    if claim_age < fra:
        if claim_age < early_age:
            claim_age = early_age  # Can't claim before early age
        years = fra - claim_age
        return max(0.7, 1 - early_reduction * years)
    elif claim_age > fra:
        years = min(claim_age - fra, late_age - fra)  # Cap at late age
        return 1 + late_bonus * years
    return 1.0


def project_spending(monthly_spend: float, spouse: bool, ret_age: int, currency_symbol: str, early_decline: float = EARLY_DECLINE, couple_decline: float = COUPLE_DECLINE, single_decline: float = SINGLE_DECLINE) -> pd.DataFrame:
    base = monthly_spend * 12  # convert to annual
    records = []
    ann_spend = base
    for age in range(ret_age, MAX_AGE + 1):
        # Phase‑specific declines
        if 62 <= age < 65:
            ann_spend *= (1 - early_decline)
        elif age >= 65:
            rate = couple_decline if spouse else single_decline
            ann_spend *= (1 - rate)
        records.append({"Age": age, "Gross_Spending": ann_spend})
    return pd.DataFrame(records)


def add_social_security(df: pd.DataFrame, ss_start: int, base_monthly: float, spouse: bool, country_config: dict) -> pd.DataFrame:
    mult = ss_multiplier(ss_start, country_config)
    annual_ss = base_monthly * 12 * mult
    if spouse:
        annual_ss *= (1 + country_config["spousal_factor"])
    df["Social_Security"] = np.where(df["Age"] >= ss_start, annual_ss, 0)
    df["Net_Need"] = np.maximum(df["Gross_Spending"] - df["Social_Security"], 0)
    return df


def pv_of_needs(df: pd.DataFrame, ret_age: int, discount_rate: float = POST_RET_ROI) -> float:
    df["Discount_Factor"] = (1 + discount_rate) ** (-(df["Age"] - ret_age))
    df["PV_Need"] = df["Net_Need"] * df["Discount_Factor"]
    return df["PV_Need"].sum()


def format_currency(amount: float, currency_symbol: str) -> str:
    """Format currency with appropriate symbol and comma separators."""
    return f"{currency_symbol}{amount:,.0f}"


def calculate_four_percent_rule(monthly_spend: float) -> float:
    """Calculate nest egg needed using traditional 4% withdrawal rule."""
    annual_spending = monthly_spend * 12
    return annual_spending * 25  # 4% rule = 25x annual expenses


def calculate_savings_balance(df: pd.DataFrame, nest_egg: float, discount_rate: float) -> pd.DataFrame:
    """Calculate year-by-year savings balance decline during retirement."""
    df = df.copy()
    savings_balance = nest_egg
    savings_balances = []
    
    for _, row in df.iterrows():
        net_need = row['Net_Need']
        
        # Apply investment return, then withdraw net need
        savings_balance = savings_balance * (1 + discount_rate) - net_need
        
        # Don't allow negative balances (would indicate plan failure)
        savings_balance = max(0, savings_balance)
        savings_balances.append(savings_balance)
    
    df['Savings_Balance'] = savings_balances
    return df


def create_retirement_chart(df: pd.DataFrame, currency: str) -> plt.Figure:
    """Create a custom matplotlib chart with dual y-axes for retirement projection."""
    # Set up the figure and primary axis
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Convert data to thousands for readability
    ages = df['Age']
    gross_spending = df['Gross_Spending'] / 1000
    social_security = df['Social_Security'] / 1000
    net_need = df['Net_Need'] / 1000
    savings_balance = df['Savings_Balance'] / 1000
    
    # Plot on left axis (spending and benefits)
    line1 = ax1.plot(ages, gross_spending, 'b-', linewidth=2, label='Gross Spending')
    line2 = ax1.plot(ages, social_security, 'orange', linewidth=2, label='Social Security')
    line3 = ax1.plot(ages, net_need, 'r-', linewidth=2, label='Net Need')
    
    # Configure left axis
    ax1.set_xlabel('Age', fontsize=12)
    ax1.set_ylabel(f'Annual Amount (thousands {currency})', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)
    
    # Create secondary y-axis for savings balance
    ax2 = ax1.twinx()
    line4 = ax2.plot(ages, savings_balance, 'g-', linewidth=3, label='Savings Balance')
    
    # Configure right axis
    ax2.set_ylabel(f'Savings Balance (thousands {currency})', fontsize=12, color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    
    # Combine legends from both axes
    lines1 = line1 + line2 + line3
    lines2 = line4
    labels1 = [l.get_label() for l in lines1]
    labels2 = [l.get_label() for l in lines2]
    
    # Create legend
    ax1.legend(lines1, labels1, loc='upper left', bbox_to_anchor=(0, 1))
    ax2.legend(lines2, labels2, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Set title and layout
    fig.suptitle('Annual Financial Projection', fontsize=14, fontweight='bold')
    fig.tight_layout()
    
    return fig


def create_interactive_retirement_chart(df: pd.DataFrame, currency: str, wealth_df: pd.DataFrame = None) -> go.Figure:
    """Create an interactive Plotly chart with dual y-axes and hover tooltips for retirement projection."""
    # Create subplot with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Convert data to thousands for readability
    ages = df['Age']
    gross_spending = df['Gross_Spending'] / 1000
    social_security = df['Social_Security'] / 1000
    net_need = df['Net_Need'] / 1000
    savings_balance = df['Savings_Balance'] / 1000
    
    # Add traces for left axis (spending and benefits)
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=gross_spending,
            mode='lines',
            name='Gross Spending',
            line=dict(color='blue', width=2),
            hovertemplate='Gross Spending: %{customdata[0]}<extra></extra>',
            customdata=[[f'{currency}{val*1000:,.0f}'] for val in gross_spending]
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=social_security,
            mode='lines',
            name='Social Security',
            line=dict(color='orange', width=2),
            hovertemplate='Social Security: %{customdata[0]}<extra></extra>',
            customdata=[[f'{currency}{val*1000:,.0f}'] for val in social_security]
        ),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=net_need,
            mode='lines',
            name='Net Need',
            line=dict(color='red', width=2),
            hovertemplate='Net Need: %{customdata[0]}<extra></extra>',
            customdata=[[f'{currency}{val*1000:,.0f}'] for val in net_need]
        ),
        secondary_y=False,
    )
    
    # Add trace for right axis (savings balance)
    fig.add_trace(
        go.Scatter(
            x=ages,
            y=savings_balance,
            mode='lines',
            name='Savings Balance',
            line=dict(color='green', width=3),
            hovertemplate='Savings Balance: %{customdata[0]}<extra></extra>',
            customdata=[[f'{currency}{val*1000:,.0f}'] for val in savings_balance]
        ),
        secondary_y=True,
    )
    
    # Add wealth accumulation trace (golden line) if provided
    if wealth_df is not None and not wealth_df.empty:
        wealth_ages = wealth_df['Age']
        wealth_balance = wealth_df['Wealth_Balance'] / 1000
        
        fig.add_trace(
            go.Scatter(
                x=wealth_ages,
                y=wealth_balance,
                mode='lines',
                name='Wealth Accumulation',
                line=dict(color='gold', width=3),
                hovertemplate='Wealth Accumulation: %{customdata[0]}<extra></extra>',
                customdata=[[f'{currency}{val*1000:,.0f}'] for val in wealth_balance]
            ),
            secondary_y=True,
        )
    
    # Update layout
    fig.update_layout(
        title='Annual Financial Projection',
        xaxis_title='Age',
        hovermode='x unified',
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Set y-axes titles
    fig.update_yaxes(title_text=f'Annual Amount (thousands {currency})', secondary_y=False)
    fig.update_yaxes(title_text=f'<span style="color:green">Savings Balance (thousands {currency})</span>', secondary_y=True)
    
    return fig


# Scenario presets
SCENARIO_PRESETS = {
    "Conservative": {
        "discount_rate": 0.04,
        "early_decline": 0.005,  # 0.5% decline ages 62-64
        "couple_decline": 0.01,   # 1% decline for couples
        "single_decline": 0.008,  # 0.8% decline for singles
        "description": "Conservative assumptions: Lower returns, slower spending decline"
    },
    "Moderate": {
        "discount_rate": 0.055,
        "early_decline": 0.0075,  # 0.75% decline ages 62-64
        "couple_decline": 0.017,   # 1.7% decline for couples
        "single_decline": 0.0125,  # 1.25% decline for singles
        "description": "Moderate assumptions: Balanced approach"
    },
    "Optimistic": {
        "discount_rate": POST_RET_ROI,
        "early_decline": EARLY_DECLINE,
        "couple_decline": COUPLE_DECLINE,
        "single_decline": SINGLE_DECLINE,
        "description": "Original optimistic assumptions: High returns, aggressive spending decline"
    }
}


def calculate_wealth_accumulation(savings_accounts: list, user_age: int, ret_age: int, inflation_rate: float) -> pd.DataFrame:
    """Calculate wealth accumulation from savings accounts over time."""
    records = []
    
    for age in range(user_age, ret_age + 1):
        years_elapsed = age - user_age
        total_wealth = 0
        
        for account in savings_accounts:
            account_value = 0
            
            # Calculate growth from initial amount
            if account['amount'] > 0:
                account_value += account['amount'] * (1 + account['roi']) ** years_elapsed
            
            # Calculate growth from monthly deposits (Future Value of Annuity)
            monthly_deposit = account.get('monthly_deposit', 0)
            if monthly_deposit > 0 and years_elapsed > 0:
                months_elapsed = years_elapsed * 12
                monthly_rate = account['roi'] / 12
                if monthly_rate > 0:
                    # FV of ordinary annuity formula
                    annuity_value = monthly_deposit * ((1 + monthly_rate) ** months_elapsed - 1) / monthly_rate
                else:
                    # If no interest, just sum the deposits
                    annuity_value = monthly_deposit * months_elapsed
                account_value += annuity_value
            
            total_wealth += account_value
        
        records.append({"Age": age, "Wealth_Balance": total_wealth})
    
    return pd.DataFrame(records)


def apply_planned_expenses(wealth_df: pd.DataFrame, planned_expenses: list, user_age: int, inflation_rate: float) -> pd.DataFrame:
    """Apply planned expenses at their scheduled ages, accounting for inflation."""
    wealth_df = wealth_df.copy()
    
    for expense in planned_expenses:
        expense_age = expense['age']
        if user_age <= expense_age <= wealth_df['Age'].max():
            # Apply inflation to expense amount
            years_to_expense = expense_age - user_age
            inflated_expense = expense['amount'] * (1 + inflation_rate) ** years_to_expense
            
            # Subtract expense from wealth at that age and all subsequent ages
            mask = wealth_df['Age'] >= expense_age
            wealth_df.loc[mask, 'Wealth_Balance'] -= inflated_expense
            
            # Ensure wealth doesn't go negative
            wealth_df['Wealth_Balance'] = wealth_df['Wealth_Balance'].clip(lower=0)
    
    return wealth_df


def calculate_net_retirement_need(required_nest_egg: float, wealth_at_retirement: float) -> dict:
    """Calculate additional savings needed after accounting for accumulated wealth."""
    additional_needed = max(0, required_nest_egg - wealth_at_retirement)
    wealth_contribution = min(required_nest_egg, wealth_at_retirement)
    
    return {
        'additional_needed': additional_needed,
        'wealth_contribution': wealth_contribution,
        'total_required': required_nest_egg
    }


def main():
    st.set_page_config(page_title="Multi-Currency Retirement Calculator", layout="wide")
    st.title("💰 Multi-Currency Retirement Needs Calculator")
    st.markdown("""Estimate how much you need to have saved to retire comfortably, based on
real‑world spending patterns and country-specific social security systems.""")
    
    # Test logging immediately when app starts
    logger.info("=== APP START ===")
    logger.info(f"Session state keys: {list(st.session_state.keys())}")
    
    # Initialize session state for wealth accumulator
    if 'savings_accounts' not in st.session_state:
        st.session_state.savings_accounts = []
    if 'planned_expenses' not in st.session_state:
        st.session_state.planned_expenses = []
    
    # Initialize session state for core input widgets
    if 'monthly_spend' not in st.session_state:
        st.session_state.monthly_spend = None
    if 'user_age' not in st.session_state:
        st.session_state.user_age = 30
    if 'spouse' not in st.session_state:
        st.session_state.spouse = True
    if 'ret_age' not in st.session_state:
        st.session_state.ret_age = 65
    if 'ss_start' not in st.session_state:
        st.session_state.ss_start = None
    if 'base_ss' not in st.session_state:
        st.session_state.base_ss = None
    
    # Initialize session state for assumption sliders
    if 'discount_rate' not in st.session_state:
        st.session_state.discount_rate = None
    if 'inflation_rate' not in st.session_state:
        st.session_state.inflation_rate = 0.02
    if 'early_decline' not in st.session_state:
        st.session_state.early_decline = None
    if 'couple_decline' not in st.session_state:
        st.session_state.couple_decline = None
    if 'single_decline' not in st.session_state:
        st.session_state.single_decline = None
    
    # Initialize session state for country tracking
    if 'selected_country' not in st.session_state:
        st.session_state.selected_country = "USA"

    with st.sidebar:
        st.header("🌍 Country & Currency")
        
        # Country selection
        selected_country = st.selectbox(
            "Select your country/region:",
            options=list(COUNTRY_CONFIG.keys()),
            index=0,
            help="This determines currency, social security rules, and default values"
        )
        
        # Detect country change
        if selected_country != st.session_state.selected_country:
            # Country changed - update currency-specific values
            country_changed = True
            st.session_state.selected_country = selected_country
        else:
            country_changed = False
        
        config = COUNTRY_CONFIG[selected_country]
        currency_symbol = config["symbol"]
        ss_name = config["ss_name"]
        
        # Update currency-specific values when country changes
        if country_changed:
            # Update monthly spend to new country default
            st.session_state.monthly_spend = config["default_monthly_spend"]
            
            # Update social security benefit to new country default  
            st.session_state.base_ss = config["default_ss_benefit"]
            
            # Show notification to user
            st.success(f"✅ Updated values for {config['currency']} currency")
        
        st.info(f"**Currency:** {config['currency']} ({currency_symbol})\n\n"
                f"**Social Security:** {ss_name}\n\n"
                f"**Full Retirement Age:** {config['ss_full_age']}")

        st.header("👤 Personal Details")
        
        user_age = st.number_input(
            "Your current age",
            min_value=18,
            max_value=80,
            value=st.session_state.user_age,
            step=1,
            help="Your current age - used to calculate inflation impact from now until retirement",
            key="user_age_input"
        )
        st.session_state.user_age = user_age

        st.header("💰 Wealth Accumulator")
        
        # Savings Accounts Section
        st.subheader("💳 Savings Accounts")
        
        # Add new savings account
        col1, col2 = st.columns([5, 1])
        with col1:
            new_account_name = st.text_input("Account name", placeholder="e.g., 401k, IRA, Savings", key="new_account_name")
        with col2:
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)  # Align button with input bottom
            if st.button("➕", help="Add account"):
                if new_account_name:
                    st.session_state.savings_accounts.append({
                        'name': new_account_name,
                        'amount': 0,
                        'roi': 0.04,
                        'monthly_deposit': 0
                    })
                    st.rerun()
        
        # Display existing savings accounts
        accounts_to_remove = []
        for i, account in enumerate(st.session_state.savings_accounts):
            with st.container():
                # Row 1: Account name and remove button (aligned with Add button)
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.text(f"🏦 {account['name']}")
                
                with col2:
                    if st.button("❌", key=f"remove_account_{i}", help="Remove account"):
                        accounts_to_remove.append(i)
                
                # Row 2: Amount input and ROI controls
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    account['amount'] = st.number_input(
                        "Amount",
                        value=account['amount'],
                        step=1000,
                        key=f"account_amount_{i}",
                        label_visibility="collapsed",
                        format="%d"
                    )
                    # Display currency symbol below the input
                    st.caption(f"{currency_symbol}{account['amount']:,}")
                
                with col2:
                    # ROI input control (matching amount input style)
                    account['roi'] = st.number_input(
                        "ROI %",
                        min_value=0.0,
                        value=account['roi'] * 100,
                        step=0.1,
                        format="%.1f",
                        key=f"account_roi_{i}",
                        label_visibility="collapsed"
                    ) / 100
                    # Display ROI caption below the input (matching amount style)
                    st.caption(f"ROI: {account['roi']*100:.1f}%")
                
                # Row 3: Monthly deposit input
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    account['monthly_deposit'] = st.number_input(
                        "Monthly Deposit",
                        value=account.get('monthly_deposit', 0),
                        step=100,
                        key=f"account_monthly_deposit_{i}",
                        label_visibility="collapsed",
                        format="%d"
                    )
                    # Display currency symbol below the input (matching amount style)
                    st.caption(f"{currency_symbol}{account['monthly_deposit']:,}/month")
                
                with col2:
                    # Empty column for alignment
                    st.empty()
                        
                st.divider()
        
        # Remove accounts marked for deletion
        for i in reversed(accounts_to_remove):
            del st.session_state.savings_accounts[i]
            st.rerun()
        
        # Planned Expenses Section
        st.subheader("🏠 Planned Expenses")
        
        # Add new planned expense
        col1, col2 = st.columns([5, 1])
        with col1:
            new_expense_name = st.text_input("Expense name", placeholder="e.g., Wedding, House, Car", key="new_expense_name")
        with col2:
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)  # Align button with input bottom
            if st.button("➕", help="Add expense"):
                if new_expense_name:
                    st.session_state.planned_expenses.append({
                        'name': new_expense_name,
                        'amount': 0,
                        'age': user_age + 5
                    })
                    st.rerun()
        
        # Display existing planned expenses
        expenses_to_remove = []
        for i, expense in enumerate(st.session_state.planned_expenses):
            with st.container():
                # Row 1: Expense name and remove button (aligned with Add button)
                col1, col2 = st.columns([5, 1])
                
                with col1:
                    st.text(f"💸 {expense['name']}")
                
                with col2:
                    if st.button("❌", key=f"remove_expense_{i}", help="Remove expense"):
                        expenses_to_remove.append(i)
                
                # Row 2: Amount input and age input
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    expense['amount'] = st.number_input(
                        "Amount",
                        value=expense['amount'],
                        step=1000,
                        key=f"expense_amount_{i}",
                        label_visibility="collapsed",
                        format="%d"
                    )
                    # Display currency symbol below the input (matching savings accounts style)
                    st.caption(f"{currency_symbol}{expense['amount']:,}")
                
                with col2:
                    # Auto-adjust expense age if it's in the past to prevent validation error
                    original_age = expense['age']
                    adjusted_expense_age = max(expense['age'], user_age)
                    was_adjusted = adjusted_expense_age != original_age
                    
                    expense['age'] = st.number_input(
                        "Age",
                        min_value=user_age,
                        max_value=80,
                        value=adjusted_expense_age,
                        step=1,
                        key=f"expense_age_{i}",
                        label_visibility="collapsed"
                    )
                    
                    # Show caption and warning if needed (matching savings accounts style)
                    if was_adjusted:
                        st.caption(f"⚠️ Adjusted from age {original_age} to {adjusted_expense_age}")
                    else:
                        st.caption(f"At age {expense['age']}")
                        
                st.divider()
        
        # Remove expenses marked for deletion
        for i in reversed(expenses_to_remove):
            del st.session_state.planned_expenses[i]
            st.rerun()

        st.header("💼 Your Financial Details")
        
        # Use country-specific defaults
        monthly_spend = st.number_input(
            f"Current household monthly spend ({currency_symbol})", 
            value=st.session_state.monthly_spend or config["default_monthly_spend"], 
            step=100,
            help=f"Your current monthly household expenses in {config['currency']}",
            key="monthly_spend_input"
        )
        st.session_state.monthly_spend = monthly_spend
        
        spouse = st.checkbox("Include spouse?", value=st.session_state.spouse, key="spouse_input")
        st.session_state.spouse = spouse
        
        ret_age = st.selectbox(
            "Desired retirement age", 
            [55, 60, 62, 65, 67, 70], 
            index=[55, 60, 62, 65, 67, 70].index(st.session_state.ret_age),
            help="Age when you want to stop working and start drawing retirement funds",
            key="ret_age_input"
        )
        st.session_state.ret_age = ret_age
        
        # Social Security claiming age with country-specific limits
        min_ss_age = config["ss_early_age"]
        max_ss_age = config["ss_late_age"]
        default_ss_age = min(max(64, min_ss_age), max_ss_age)
        
        ss_start = st.slider(
            f"Start {ss_name} at age", 
            min_value=min_ss_age, 
            max_value=max_ss_age, 
            value=st.session_state.ss_start or default_ss_age,
            help=f"Age when you claim {ss_name} benefits. Earlier = lower benefits, later = higher benefits",
            key="ss_start_input"
        )
        st.session_state.ss_start = ss_start
        
        base_ss = st.number_input(
            f"Estimated individual {ss_name} benefit at age {config['ss_full_age']} ({currency_symbol}/month)", 
            value=st.session_state.base_ss or config["default_ss_benefit"], 
            step=50,
            help=f"Your estimated monthly {ss_name} benefit at full retirement age",
            key="base_ss_input"
        )
        st.session_state.base_ss = base_ss

        # Show country-specific information
        st.header("📊 Country Details")
        
        # Social Security multiplier info
        current_mult = ss_multiplier(ss_start, config)
        if spouse:
            spousal_info = f" + {config['spousal_factor']*100:.0f}% spousal benefit"
        else:
            spousal_info = ""
            
        st.metric(
            f"{ss_name} Benefit Multiplier",
            f"{current_mult:.1%}{spousal_info}",
            help=f"Your benefit adjustment for claiming at age {ss_start}"
        )

        # Advanced assumptions section
        st.header("⚙️ Advanced Assumptions")
        
        # Scenario selection
        scenario = st.selectbox(
            "Planning Scenario",
            options=list(SCENARIO_PRESETS.keys()),
            index=1,  # Default to "Moderate"
            help="Choose conservative, moderate, or optimistic assumptions"
        )
        
        preset = SCENARIO_PRESETS[scenario]
        st.info(preset["description"])
        
        # Start with preset values
        discount_rate = preset["discount_rate"]
        early_decline = preset["early_decline"]
        couple_decline = preset["couple_decline"]
        single_decline = preset["single_decline"]
        
        # Default inflation rate (can be overridden in custom adjustments)
        inflation_rate = 0.02  # 2% default
        
        # Allow custom adjustments
        with st.expander("🔧 Customize Assumptions"):
            st.warning("⚠️ **Caution**: The default 'Optimistic' scenario may underestimate retirement needs. Consider using 'Conservative' or 'Moderate' for safer planning.")
            
            discount_rate = st.slider(
                "Real investment return during retirement",
                min_value=0.03,
                max_value=0.08,
                value=st.session_state.discount_rate or preset["discount_rate"],
                step=0.005,
                format="%.1f%%",
                help="Higher returns reduce needed savings but increase risk. Conservative: 3-4%, Moderate: 5-6%, Optimistic: 7%+",
                key="discount_rate_slider"
            )
            st.session_state.discount_rate = discount_rate
            
            inflation_rate = st.slider(
                "Expected inflation rate", 
                min_value=0.0, 
                max_value=0.08, 
                value=st.session_state.inflation_rate,
                step=0.001,
                format="%.1f%%",
                help="Expected annual inflation rate from now until retirement. This will increase your spending needs over time.",
                key="inflation_rate_slider"
            )
            st.session_state.inflation_rate = inflation_rate
            
            st.subheader("Spending Decline Rates")
            st.caption("Research shows spending typically declines in retirement, but rates vary widely.")
            
            early_decline = st.slider(
                "Spending decline ages 62-64 (% per year)",
                min_value=0.0,
                max_value=0.02,
                value=st.session_state.early_decline or preset["early_decline"],
                step=0.0025,
                format="%.2f%%",
                help="Transition from 'go-go' to 'slow-go' years",
                key="early_decline_slider"
            )
            st.session_state.early_decline = early_decline
            
            couple_decline = st.slider(
                "Couple spending decline after 65 (% per year)",
                min_value=0.005,
                max_value=0.03,
                value=st.session_state.couple_decline or preset["couple_decline"],
                step=0.0025,
                format="%.2f%%",
                help="Annual spending decline for couples in later retirement",
                key="couple_decline_slider"
            )
            st.session_state.couple_decline = couple_decline
            
            single_decline = st.slider(
                "Single spending decline after 65 (% per year)",
                min_value=0.005,
                max_value=0.025,
                value=st.session_state.single_decline or preset["single_decline"],
                step=0.0025,
                format="%.2f%%",
                help="Annual spending decline for singles in later retirement",
                key="single_decline_slider"
            )
            st.session_state.single_decline = single_decline

    # Validate user age is less than retirement age
    if user_age >= ret_age:
        st.error(f"⚠️ Your current age ({user_age}) must be less than your desired retirement age ({ret_age}).")
        st.stop()
    
    # Calculate inflation-adjusted spending
    years_to_retirement = ret_age - user_age
    inflated_monthly_spend = monthly_spend * (1 + inflation_rate) ** years_to_retirement
    
    # Log all components of inflation calculation
    logger.info("=== INFLATION CALCULATION COMPONENTS ===")
    logger.info(f"monthly_spend (raw): {monthly_spend}")
    logger.info(f"inflation_rate: {inflation_rate}")
    logger.info(f"user_age: {user_age}")
    logger.info(f"ret_age: {ret_age}")
    logger.info(f"years_to_retirement: {years_to_retirement}")
    logger.info(f"inflation_multiplier: {(1 + inflation_rate) ** years_to_retirement}")
    logger.info(f"inflated_monthly_spend: {inflated_monthly_spend}")
    
    # Calculate wealth accumulation from savings accounts and expenses
    wealth_df = pd.DataFrame()
    wealth_at_retirement = 0
    
    if st.session_state.savings_accounts:
        wealth_df = calculate_wealth_accumulation(st.session_state.savings_accounts, user_age, ret_age, inflation_rate)
        wealth_df = apply_planned_expenses(wealth_df, st.session_state.planned_expenses, user_age, inflation_rate)
        wealth_at_retirement = wealth_df[wealth_df['Age'] == ret_age]['Wealth_Balance'].iloc[0] if len(wealth_df) > 0 else 0
    
    # Log wealth calculation results
    logger.info("=== WEALTH CALCULATION ===")
    logger.info(f"Savings accounts: {len(st.session_state.savings_accounts)}")
    for i, acc in enumerate(st.session_state.savings_accounts):
        logger.info(f"  Account {i}: {acc['name']} - Amount: {acc['amount']}, ROI: {acc['roi']}")
    logger.info(f"Planned expenses: {len(st.session_state.planned_expenses)}")
    for i, exp in enumerate(st.session_state.planned_expenses):
        logger.info(f"  Expense {i}: {exp['name']} - Amount: {exp['amount']}, Age: {exp['age']}")
    logger.info(f"wealth_at_retirement: {wealth_at_retirement:,.2f}")
    
    # Calculations with custom parameters (using inflation-adjusted spending)
    # Log spending and social security inputs
    logger.info("=== SPENDING/SS CALCULATION INPUTS ===")
    logger.info(f"inflated_monthly_spend: {inflated_monthly_spend}")
    logger.info(f"spouse: {spouse}")
    logger.info(f"early_decline: {early_decline}")
    logger.info(f"couple_decline: {couple_decline}")
    logger.info(f"single_decline: {single_decline}")
    logger.info(f"ss_start: {ss_start}")
    logger.info(f"base_ss: {base_ss}")
    
    df = project_spending(inflated_monthly_spend, spouse, ret_age, currency_symbol, early_decline, couple_decline, single_decline)
    df = add_social_security(df, ss_start, base_ss, spouse, config)
    
    # Log inputs to nest_egg calculation
    logger.info("=== NEST EGG CALCULATION INPUTS ===")
    logger.info(f"ret_age: {ret_age}")
    logger.info(f"discount_rate: {discount_rate}")
    logger.info(f"df shape: {df.shape}")
    logger.info(f"df.head(): {df.head().to_dict()}")
    
    nest_egg = pv_of_needs(df, ret_age, discount_rate)
    
    # Calculate net retirement need accounting for accumulated wealth
    retirement_analysis = calculate_net_retirement_need(nest_egg, wealth_at_retirement)
    
    # Log retirement analysis results
    logger.info("=== RETIREMENT ANALYSIS ===")
    logger.info(f"nest_egg: {nest_egg:,.2f}")
    logger.info(f"additional_needed: {retirement_analysis['additional_needed']:,.2f}")
    logger.info(f"wealth_contribution: {retirement_analysis['wealth_contribution']:,.2f}")
    logger.info(f"total_required: {retirement_analysis['total_required']:,.2f}")
    logger.info(f"UI condition (wealth_at_retirement > 0): {wealth_at_retirement > 0}")
    
    # Add savings balance calculation (using total required nest egg)
    df = calculate_savings_balance(df, nest_egg, discount_rate)
    
    # Calculate 4% rule comparison (using inflation-adjusted spending)
    four_percent_rule = calculate_four_percent_rule(inflated_monthly_spend)

    # Main display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # Show different metrics based on whether user has wealth accumulation
        if wealth_at_retirement > 0:
            # Log which UI branch is shown
            logger.info("=== UI DISPLAY: Wealth Accumulation Branch ===")
            logger.info(f"Displaying 'Additional savings needed': {retirement_analysis['additional_needed']:,.2f}")
            logger.info(f"Displaying 'Total retirement need': {retirement_analysis['total_required']:,.2f}")
            st.metric(
                label=f"**Additional savings needed by age {ret_age}**", 
                value=format_currency(retirement_analysis['additional_needed'], currency_symbol),
                help="Additional savings needed after accounting for your current wealth accumulation"
            )
            
            st.metric(
                label=f"**Wealth contribution**", 
                value=format_currency(retirement_analysis['wealth_contribution'], currency_symbol),
                help="Amount your current savings and planned expenses will contribute toward retirement"
            )
            
            st.metric(
                label=f"**Total retirement need**", 
                value=format_currency(retirement_analysis['total_required'], currency_symbol),
                help="Total nest egg required before considering existing wealth"
            )
        else:
            # Log which UI branch is shown
            logger.info("=== UI DISPLAY: Basic Branch ===")
            logger.info(f"Displaying 'Required nest-egg': {nest_egg:,.2f}")
            st.metric(
                label=f"**Required nest‑egg at age {ret_age}**", 
                value=format_currency(nest_egg, currency_symbol),
                help="Present value of your retirement needs minus Social Security benefits"
            )
        
        # 4% rule comparison
        comparison_ratio = nest_egg / four_percent_rule if four_percent_rule > 0 else 0
        if comparison_ratio < 0.7:
            comparison_delta = f"-{(1-comparison_ratio)*100:.0f}% vs 4% rule"
            comparison_color = "normal"
        elif comparison_ratio > 1.3:
            comparison_delta = f"+{(comparison_ratio-1)*100:.0f}% vs 4% rule"
            comparison_color = "inverse"
        else:
            comparison_delta = f"{(comparison_ratio-1)*100:+.0f}% vs 4% rule"
            comparison_color = "off"
            
        st.metric(
            "4% Rule Comparison",
            format_currency(four_percent_rule, currency_symbol),
            delta=comparison_delta,
            help="Traditional 4% withdrawal rule suggests 25x annual expenses. Significant differences may indicate aggressive assumptions."
        )
        
        # Inflation impact metric
        inflation_multiplier = (1 + inflation_rate) ** years_to_retirement
        st.metric(
            "Inflation Impact",
            f"{inflation_multiplier:.1f}x",
            delta=f"{format_currency(inflated_monthly_spend - monthly_spend, currency_symbol)}/month",
            help=f"Due to {inflation_rate:.1%} annual inflation over {years_to_retirement} years, your spending needs will be {inflation_multiplier:.1f}x higher at retirement"
        )
        
        # Additional metrics
        total_ss_pv = (df["Social_Security"] * df["Discount_Factor"]).sum()
        total_spending_pv = (df["Gross_Spending"] * df["Discount_Factor"]).sum()
        
        st.metric(
            f"Total {ss_name} (Present Value)",
            format_currency(total_ss_pv, currency_symbol),
            help=f"Present value of all {ss_name} benefits you'll receive"
        )
        
        st.metric(
            "Social Security Coverage",
            f"{(total_ss_pv / total_spending_pv * 100):.1f}%",
            help="Percentage of retirement expenses covered by Social Security"
        )
        
        # Show key assumptions being used
        st.subheader("📋 Current Assumptions")
        st.text(f"• Investment return: {discount_rate:.1%}")
        st.text(f"• Inflation rate: {inflation_rate:.1%}/year")
        st.text(f"• Years to retirement: {years_to_retirement}")
        st.text(f"• Early decline (62-64): {early_decline:.2%}/year")
        if spouse:
            st.text(f"• Couple decline (65+): {couple_decline:.2%}/year")
        else:
            st.text(f"• Single decline (65+): {single_decline:.2%}/year")
        
        # Show wealth accumulation summary
        if st.session_state.savings_accounts or st.session_state.planned_expenses:
            st.subheader("💰 Wealth Summary")
            current_total = sum(account['amount'] for account in st.session_state.savings_accounts)
            total_expenses = sum(expense['amount'] for expense in st.session_state.planned_expenses)
            st.text(f"• Current savings: {format_currency(current_total, currency_symbol)}")
            st.text(f"• Planned expenses: {format_currency(total_expenses, currency_symbol)}")
            st.text(f"• Wealth at retirement: {format_currency(wealth_at_retirement, currency_symbol)}")
            if wealth_at_retirement > 0:
                coverage_pct = (wealth_at_retirement / nest_egg * 100) if nest_egg > 0 else 0
                st.text(f"• Retirement coverage: {coverage_pct:.1f}%")

    with col2:
        # Create interactive dual-axis chart with hover tooltips
        st.subheader("Annual Financial Projection")
        if wealth_df is not None and not wealth_df.empty:
            st.caption("📊 **Interactive Chart**: Hover over any point to see detailed values | **Left axis**: Gross Spending (blue), Social Security (orange), Net Need (red) | **Right axis**: **Savings Balance (green), Wealth Accumulation (gold)**")
        else:
            st.caption("📊 **Interactive Chart**: Hover over any point to see detailed values | **Left axis**: Gross Spending (blue), Social Security (orange), Net Need (red) | **Right axis**: **Savings Balance (green)**")
        
        # Generate and display the interactive chart
        fig = create_interactive_retirement_chart(df, config['symbol'], wealth_df)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add interpretation help
        final_balance = df.iloc[-1]['Savings_Balance']
        if final_balance > nest_egg * 0.5:
            st.success(f"✅ **Sustainable plan**: Savings remain strong ({config['symbol']}{final_balance/1000:.0f}k at age 95)")
        elif final_balance > 0:
            st.warning(f"⚠️ **Adequate plan**: Savings decline but remain positive ({config['symbol']}{final_balance/1000:.0f}k at age 95)")
        else:
            st.error("🚨 **Unsustainable plan**: Savings would be depleted before age 95")
            # Find when savings run out
            depletion_age = None
            for i, balance in enumerate(df['Savings_Balance']):
                if balance <= 0:
                    depletion_age = df.iloc[i]['Age']
                    break
            if depletion_age:
                st.error(f"💸 Savings would run out around age {depletion_age:.0f}")

    # Detailed breakdown
    st.subheader("📋 Annual Spending & Income Detail (first 20 years)")
    
    # Format the dataframe for display
    display_df = df[["Age", "Gross_Spending", "Social_Security", "Net_Need", "Savings_Balance"]].head(20).copy()
    
    # Format currency columns
    for col in ["Gross_Spending", "Social_Security", "Net_Need", "Savings_Balance"]:
        display_df[col] = display_df[col].apply(lambda x: format_currency(x, currency_symbol))
    
    # Rename columns for display
    display_df.columns = ["Age", f"Gross Spending ({config['currency']})", 
                         f"{ss_name} ({config['currency']})", f"Net Need ({config['currency']})",
                         f"Savings Balance ({config['currency']})"]
    
    st.dataframe(display_df, use_container_width=True)

    # Country-specific notes
    st.subheader("🏛️ Country-Specific Information")
    
    country_notes = {
        "USA": """
        **🇺🇸 United States Social Security:**
        - Full Retirement Age: 67 for those born 1960+
        - Early claiming: Available from age 62 with 30% reduction at minimum
        - Delayed retirement credits: 8% per year until age 70
        - Spousal benefits: Up to 50% of worker's benefit
        - Benefits based on highest 35 years of earnings
        """,
        "Europe": """
        **🇪🇺 European Union (Average):**
        - Retirement ages vary: 60-67 across EU countries
        - Average pension: ~€1,345/month (2022 data)
        - Benefits vary significantly by country (€226 Bulgaria to €2,575 Luxembourg)
        - Most systems are pay-as-you-go with earnings-related benefits
        - Spousal benefits and rules vary by country
        """,
        "UK": """
        **🇬🇧 United Kingdom State Pension:**
        - State Pension age: Currently 66, rising to 67 by 2028
        - Full pension: £230.25/week (~£962/month) for 2025-26
        - No early claiming available - must wait until State Pension age
        - Individual entitlement - no spousal benefits from State Pension
        - Requires 35 years of National Insurance contributions for full pension
        """,
        "Israel": """
        **🇮🇱 Israel Old Age Pension (Bituach Leumi):**
        - Retirement age: 67 for men, transitioning to 67 for women
        - Basic pension: ₪1,795/month for individual (2025)
        - Spousal increment: ₪902 additional
        - Seniority increment: 2% per year of contributions (max 50%)
        - Minimum 10 years contributions required, full benefits at 35+ years
        """
    }
    
    st.markdown(country_notes[selected_country])

    # Risk warnings and scenario comparison
    st.subheader("⚠️ Important Planning Considerations")
    
    # Show scenario comparison
    col_scenario1, col_scenario2, col_scenario3 = st.columns(3)
    
    scenarios_to_compare = ["Conservative", "Moderate", "Optimistic"]
    scenario_results = {}
    
    for scenario_name in scenarios_to_compare:
        scenario_preset = SCENARIO_PRESETS[scenario_name]
        temp_df = project_spending(inflated_monthly_spend, spouse, ret_age, currency_symbol, 
                                 scenario_preset["early_decline"], scenario_preset["couple_decline"], scenario_preset["single_decline"])
        temp_df = add_social_security(temp_df, ss_start, base_ss, spouse, config)
        temp_nest_egg = pv_of_needs(temp_df, ret_age, scenario_preset["discount_rate"])
        temp_df = calculate_savings_balance(temp_df, temp_nest_egg, scenario_preset["discount_rate"])
        scenario_results[scenario_name] = temp_nest_egg
    
    with col_scenario1:
        st.metric("Conservative Scenario", format_currency(scenario_results["Conservative"], currency_symbol),
                 help="Lower investment returns (4%), slower spending decline")
    
    with col_scenario2:
        st.metric("Moderate Scenario", format_currency(scenario_results["Moderate"], currency_symbol),
                 help="Balanced assumptions (5.5% returns, moderate spending decline)")
    
    with col_scenario3:
        st.metric("Optimistic Scenario", format_currency(scenario_results["Optimistic"], currency_symbol),
                 help="Original assumptions (7% returns, aggressive spending decline)")
    
    if scenario == "Optimistic":
        st.warning("🚨 **You're using optimistic assumptions!** Consider reviewing the 'Conservative' or 'Moderate' scenarios for more robust planning.")
    elif scenario == "Conservative":
        st.success("✅ **Conservative planning approach** - This provides a safety buffer for retirement planning.")
    else:
        st.info("ℹ️ **Balanced approach** - Moderate assumptions balance optimism with prudent planning.")

    # Methodology and assumptions
    with st.expander("📖 Methodology & Data Sources"):
        st.markdown(f"""
        **⚠️ IMPORTANT DISCLAIMERS:**
        - Results vary significantly based on assumptions. Conservative planning is recommended.
        - Healthcare costs, inflation, and sequence-of-returns risk are not fully modeled.
        - Past investment performance doesn't guarantee future results.
        - Consider consulting a financial advisor for personalized guidance.
        
        **Spending Decline Research:**
        - Based on RAND 2023, CRR 2021, J.P. Morgan Guide to Retirement 2025
        - **Original research findings** (used in "Optimistic" scenario):
          • 1% annual decline ages 62-64 (go-go to slow-go transition)
          • Couples: 2.4% annual decline after 65
          • Singles: 1.7% annual decline after 65
        - **Reality check**: Individual spending patterns vary widely and may not decline as much
        
        **Investment Return Assumptions:**
        - **Conservative**: 4% real return (bond-heavy portfolio)
        - **Moderate**: 5.5% real return (balanced portfolio)
        - **Optimistic**: 7% real return (stock-heavy portfolio, based on historical averages)
        - Does not account for sequence-of-returns risk (poor early returns can be devastating)
        
        **4% Rule Comparison:**
        - Traditional safe withdrawal rate of 4% annually (25x expenses)
        - Based on Trinity Study and subsequent research
        - Provides benchmark for reality-checking results
        
        **Savings Balance Projection (Green Line):**
        - Shows your actual savings balance declining over retirement years
        - Each year: balance grows by investment return, then net need is withdrawn
        - **Interpretation**:
          • Line goes up: Your assumptions may be too optimistic (high returns/aggressive spending decline)  
          • Line declines gradually: Reasonable plan that depletes savings by age 95
          • Line hits zero: Plan fails - savings run out before end of retirement
        - Provides reality check on whether your retirement plan is sustainable
        
        **Social Security Data Sources:**
        - **USA:** Social Security Administration (SSA) 2024 data
        - **Europe:** Eurostat 2022, average EU pension expenditure
        - **UK:** Gov.UK 2025-26 State Pension rates
        - **Israel:** National Insurance Institute (Bituach Leumi) 2025 rates
        
        **Currency & Economic Assumptions:**
        - All figures in nominal terms for selected currency
        - Social Security benefits assumed to keep pace with inflation
        - Spending patterns assumed consistent across countries (may vary in practice)
        """)

    st.caption(f"© 2025 • Country data from official government sources • Calculations in {config['currency']} ({currency_symbol})")


if __name__ == "__main__":
    main()
