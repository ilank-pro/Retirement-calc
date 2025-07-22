import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


def project_spending(monthly_spend: float, spouse: bool, ret_age: int, currency_symbol: str) -> pd.DataFrame:
    base = monthly_spend * 12  # convert to annual
    records = []
    ann_spend = base
    for age in range(ret_age, MAX_AGE + 1):
        # Phase‑specific declines
        if 62 <= age < 65:
            ann_spend *= (1 - EARLY_DECLINE)
        elif age >= 65:
            rate = COUPLE_DECLINE if spouse else SINGLE_DECLINE
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


def pv_of_needs(df: pd.DataFrame, ret_age: int) -> float:
    df["Discount_Factor"] = (1 + POST_RET_ROI) ** (-(df["Age"] - ret_age))
    df["PV_Need"] = df["Net_Need"] * df["Discount_Factor"]
    return df["PV_Need"].sum()


def format_currency(amount: float, currency_symbol: str) -> str:
    """Format currency with appropriate symbol and comma separators."""
    return f"{currency_symbol}{amount:,.0f}"


def main():
    st.set_page_config(page_title="Multi-Currency Retirement Calculator", layout="wide")
    st.title("💰 Multi-Currency Retirement Needs Calculator")
    st.markdown("""Estimate how much you need to have saved to retire comfortably, based on
real‑world spending patterns and country-specific social security systems.""")

    with st.sidebar:
        st.header("🌍 Country & Currency")
        
        # Country selection
        selected_country = st.selectbox(
            "Select your country/region:",
            options=list(COUNTRY_CONFIG.keys()),
            index=0,
            help="This determines currency, social security rules, and default values"
        )
        
        config = COUNTRY_CONFIG[selected_country]
        currency_symbol = config["symbol"]
        ss_name = config["ss_name"]
        
        st.info(f"**Currency:** {config['currency']} ({currency_symbol})\n\n"
                f"**Social Security:** {ss_name}\n\n"
                f"**Full Retirement Age:** {config['ss_full_age']}")

        st.header("💼 Your Financial Details")
        
        # Use country-specific defaults
        monthly_spend = st.number_input(
            f"Current household monthly spend ({currency_symbol})", 
            value=config["default_monthly_spend"], 
            step=100,
            help=f"Your current monthly household expenses in {config['currency']}"
        )
        
        spouse = st.checkbox("Include spouse?", value=True)
        
        ret_age = st.selectbox(
            "Desired retirement age", 
            [55, 60, 62, 65, 67, 70], 
            index=3,
            help="Age when you want to stop working and start drawing retirement funds"
        )
        
        # Social Security claiming age with country-specific limits
        min_ss_age = config["ss_early_age"]
        max_ss_age = config["ss_late_age"]
        default_ss_age = min(max(64, min_ss_age), max_ss_age)
        
        ss_start = st.slider(
            f"Start {ss_name} at age", 
            min_value=min_ss_age, 
            max_value=max_ss_age, 
            value=default_ss_age,
            help=f"Age when you claim {ss_name} benefits. Earlier = lower benefits, later = higher benefits"
        )
        
        base_ss = st.number_input(
            f"Estimated individual {ss_name} benefit at age {config['ss_full_age']} ({currency_symbol}/month)", 
            value=config["default_ss_benefit"], 
            step=50,
            help=f"Your estimated monthly {ss_name} benefit at full retirement age"
        )

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

    # Calculations
    df = project_spending(monthly_spend, spouse, ret_age, currency_symbol)
    df = add_social_security(df, ss_start, base_ss, spouse, config)
    nest_egg = pv_of_needs(df, ret_age)

    # Main display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric(
            label=f"**Required nest‑egg at age {ret_age}**", 
            value=format_currency(nest_egg, currency_symbol),
            help="Present value of your retirement needs minus Social Security benefits"
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

    with col2:
        # Create the chart with proper currency formatting
        chart_df = df.set_index("Age")[["Gross_Spending", "Social_Security", "Net_Need"]]
        
        # Convert to thousands for better readability
        chart_df = chart_df / 1000
        
        st.subheader(f"Annual Financial Projection (thousands {config['currency']})")
        st.line_chart(chart_df)

    # Detailed breakdown
    st.subheader("📋 Annual Spending & Income Detail (first 20 years)")
    
    # Format the dataframe for display
    display_df = df[["Age", "Gross_Spending", "Social_Security", "Net_Need"]].head(20).copy()
    
    # Format currency columns
    for col in ["Gross_Spending", "Social_Security", "Net_Need"]:
        display_df[col] = display_df[col].apply(lambda x: format_currency(x, currency_symbol))
    
    # Rename columns for display
    display_df.columns = ["Age", f"Gross Spending ({config['currency']})", 
                         f"{ss_name} ({config['currency']})", f"Net Need ({config['currency']})"]
    
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

    # Methodology and assumptions
    with st.expander("📖 Methodology & Data Sources"):
        st.markdown(f"""
        **Spending Decline Assumptions:**
        - Based on RAND 2023, CRR 2021, J.P. Morgan Guide to Retirement 2025
        - 1% annual decline ages 62-64 (go-go to slow-go transition)
        - Couples: 2.4% annual decline after 65
        - Singles: 1.7% annual decline after 65
        
        **Investment Return:**
        - 7% real return assumption during retirement
        - Based on historical long-term market performance
        
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
