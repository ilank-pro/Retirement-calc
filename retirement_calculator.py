import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging
import sys
from datetime import datetime
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.legends import Legend
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.fonts import addMapping
import platform
import os
from bidi.algorithm import get_display
import re

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


def export_chart_for_pdf(df: pd.DataFrame, currency: str, wealth_df: pd.DataFrame = None) -> BytesIO:
    """Create a matplotlib chart optimized for PDF export."""
    # Create the chart using existing function
    fig = create_retirement_chart(df, currency)
    
    # Save to BytesIO for embedding in PDF
    img_buffer = BytesIO()
    fig.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
    img_buffer.seek(0)
    plt.close(fig)  # Clean up memory
    
    return img_buffer


def contains_hebrew(text):
    """Check if text contains Hebrew characters."""
    if not text:
        return False
    hebrew_pattern = re.compile(r'[\u0590-\u05FF]')
    return bool(hebrew_pattern.search(str(text)))

def process_hebrew_text(text):
    """Process Hebrew text for proper RTL display in PDF."""
    if not text:
        return text
    
    text_str = str(text).strip()
    
    # If text contains Hebrew characters, apply BiDi processing for proper RTL display
    if contains_hebrew(text_str):
        try:
            # Apply bidirectional text processing for proper RTL display
            # This converts from logical order (typing order) to visual order (display order)
            processed_text = get_display(text_str)
            if DEBUG_LOGGING:
                logger.info(f"Hebrew text processed: '{text_str}' -> '{processed_text}'")
            return processed_text
        except Exception as e:
            if DEBUG_LOGGING:
                logger.error(f"BiDi processing failed for '{text_str}': {e}")
            return text_str
    else:
        # Return non-Hebrew text as-is
        return text_str

def get_appropriate_font(text, hebrew_font_name='HebrewFont', default_font='Helvetica'):
    """Get appropriate font based on text content."""
    if contains_hebrew(str(text)):
        return hebrew_font_name
    else:
        return default_font

def register_hebrew_fonts():
    """Register Hebrew-capable fonts for PDF generation with cross-platform support."""
    try:
        system = platform.system().lower()
        if DEBUG_LOGGING:
            logger.info(f"=== HEBREW FONT REGISTRATION START ===")
            logger.info(f"Operating system: {system}")
        
        # Define potential font paths for different operating systems
        font_paths = {
            'darwin': [  # macOS - prioritize fonts that handle Hebrew RTL properly
                '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
                '/System/Library/Fonts/Supplemental/NewPeninimMT.ttc',
                '/System/Library/Fonts/SFHebrew.ttf',
                '/System/Library/Fonts/Supplemental/Arial.ttf',
                '/System/Library/Fonts/Helvetica.ttc',
                '/Library/Fonts/Arial.ttf',
                '/System/Library/Fonts/Times.ttc'
            ],
            'windows': [  # Windows
                'C:/Windows/Fonts/arial.ttf',
                'C:/Windows/Fonts/arialuni.ttf',
                'C:/Windows/Fonts/times.ttf',
                'C:/Windows/Fonts/calibri.ttf'
            ],
            'linux': [  # Linux
                '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
                '/usr/share/fonts/TTF/DejaVuSans.ttf',
                '/usr/share/fonts/truetype/arial.ttf'
            ]
        }
        
        # Get appropriate font paths for current system
        if system == 'darwin':
            candidate_fonts = font_paths['darwin']
        elif system == 'windows':
            candidate_fonts = font_paths['windows']
        else:  # Linux and others
            candidate_fonts = font_paths['linux']
        
        # Try to register the first available font
        hebrew_font_registered = False
        hebrew_bold_font_registered = False
        
        if DEBUG_LOGGING:
            logger.info(f"Trying to register fonts from: {candidate_fonts}")
        
        for font_path in candidate_fonts:
            if DEBUG_LOGGING:
                logger.info(f"Checking font path: {font_path}")
                logger.info(f"Font exists: {os.path.exists(font_path)}")
            
            if os.path.exists(font_path):
                try:
                    # Register regular font
                    if not hebrew_font_registered:
                        pdfmetrics.registerFont(TTFont('HebrewFont', font_path))
                        hebrew_font_registered = True
                        if DEBUG_LOGGING:
                            logger.info(f"✅ Successfully registered Hebrew font: {font_path}")
                    
                    # Try to find bold variant
                    if not hebrew_bold_font_registered:
                        bold_path = font_path.replace('.ttf', '-Bold.ttf').replace('.ttc', '-Bold.ttc')
                        if os.path.exists(bold_path):
                            pdfmetrics.registerFont(TTFont('HebrewFont-Bold', bold_path))
                            hebrew_bold_font_registered = True
                            if DEBUG_LOGGING:
                                logger.info(f"✅ Successfully registered Hebrew bold font: {bold_path}")
                        else:
                            # Use regular font for bold if bold variant not found
                            pdfmetrics.registerFont(TTFont('HebrewFont-Bold', font_path))
                            hebrew_bold_font_registered = True
                            if DEBUG_LOGGING:
                                logger.info(f"✅ Using regular font for bold: {font_path}")
                    
                    if hebrew_font_registered and hebrew_bold_font_registered:
                        break
                        
                except Exception as e:
                    if DEBUG_LOGGING:
                        logger.error(f"❌ Failed to register font {font_path}: {str(e)}")
                    # Continue to next font if this one fails
                    continue
        
        # Create font family mapping if fonts were registered
        if hebrew_font_registered:
            addMapping('HebrewFont', 0, 0, 'HebrewFont')  # normal
            if hebrew_bold_font_registered:
                addMapping('HebrewFont', 1, 0, 'HebrewFont-Bold')  # bold
            else:
                addMapping('HebrewFont', 1, 0, 'HebrewFont')  # bold fallback to normal
            
            if DEBUG_LOGGING:
                logger.info(f"🎯 Font family mapping created. Returning: HebrewFont")
            return 'HebrewFont'  # Return the registered font name
        
        # Fallback to built-in fonts if no Hebrew fonts found
        if DEBUG_LOGGING:
            logger.warning(f"⚠️  No Hebrew fonts found, falling back to Helvetica")
        return 'Helvetica'
        
    except Exception as e:
        if DEBUG_LOGGING:
            logger.error(f"❌ Exception in font registration: {str(e)}")
        # Ultimate fallback to built-in font
        return 'Helvetica'


def generate_pdf_report(
    retirement_analysis: dict,
    wealth_at_retirement: float,
    nest_egg: float,
    df: pd.DataFrame,
    display_df: pd.DataFrame,
    config: dict,
    user_age: int,
    ret_age: int,
    spouse: bool,
    monthly_spend: float,
    ss_start: int,
    base_ss: float,
    discount_rate: float,
    inflation_rate: float,
    early_decline: float,
    couple_decline: float,
    single_decline: float,
    years_to_retirement: int,
    savings_accounts: list,
    planned_expenses: list,
    scenario: str,
    wealth_df: pd.DataFrame = None
) -> bytes:
    """Generate a comprehensive PDF report of the retirement analysis."""
    
    if DEBUG_LOGGING:
        logger.info(f"🎯 PDF GENERATION START")
        logger.info(f"📊 Received savings_accounts: {savings_accounts}")
        logger.info(f"📊 Received planned_expenses: {planned_expenses}")
    
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72,
                           topMargin=72, bottomMargin=18)
    
    # Register Hebrew-capable fonts and use them
    try:
        hebrew_font_name = register_hebrew_fonts()
        if DEBUG_LOGGING:
            logger.info(f"📄 PDF Generation using font: {hebrew_font_name}")
    except Exception as e:
        if DEBUG_LOGGING:
            logger.error(f"❌ Font registration failed in PDF generation: {str(e)}")
        hebrew_font_name = 'Helvetica'  # Fallback to Helvetica if registration fails
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='Helvetica',
        fontSize=18,
        textColor=colors.navy,
        spaceAfter=30,
        alignment=1  # Center alignment
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontName='Helvetica',
        fontSize=14,
        textColor=colors.darkblue,
        spaceAfter=12
    )
    
    # Create a normal style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10
    )
    
    # Build the document content
    story = []
    
    # Title and header
    story.append(Paragraph("Multi-Currency Retirement Needs Calculator", title_style))
    story.append(Paragraph('Analysis Report', heading_style))
    story.append(Paragraph(f'Generated: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}', normal_style))
    story.append(Spacer(1, 20))
    
    # Personal Information Section (simplified)
    story.append(Paragraph("Personal Information", heading_style))
    story.append(Paragraph(f'Current Age: {user_age} years', normal_style))
    story.append(Paragraph(f'Retirement Age: {ret_age} years', normal_style))
    story.append(Paragraph(f'Spouse: {"Yes" if spouse else "No"}', normal_style))
    story.append(Spacer(1, 20))
    
    # Key Metrics Section
    story.append(Paragraph("Key Financial Metrics", heading_style))
    
    # Format currency values
    currency_symbol = config.get('symbol', '$')
    
    if wealth_at_retirement > 0:
        additional_needed = retirement_analysis["additional_needed"]
        total_required = retirement_analysis["total_required"]
        metrics_data = [
            ['Additional Savings Needed:', f'{currency_symbol}{additional_needed:,.0f}'],
            ['Total Expected Savings:', f'{currency_symbol}{wealth_at_retirement:,.0f}'],
            ['Total Retirement Need:', f'{currency_symbol}{total_required:,.0f}'],
            ['Current Monthly Spending:', f'{currency_symbol}{monthly_spend:,.0f}/month']
        ]
    else:
        metrics_data = [
            ['Required Nest Egg:', f'{currency_symbol}{nest_egg:,.0f}'],
            ['Current Monthly Spending:', f'{currency_symbol}{monthly_spend:,.0f}/month']
        ]
    
    metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
    table_style = TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    metrics_table.setStyle(table_style)
    story.append(metrics_table)
    story.append(Spacer(1, 20))
    
    # Assumptions Section
    story.append(Paragraph("Planning Assumptions", heading_style))
    assumptions_data = [
        ['Investment Return (Real):', f'{discount_rate:.1%}'],
        ['Inflation Rate:', f'{inflation_rate:.1%}/year'],
        ['Early Spending Decline (62-64):', f'{early_decline:.2%}/year'],
        ['Later Spending Decline (65+):', f'{(couple_decline if spouse else single_decline):.2%}/year'],
        ['Social Security Start Age:', f'{ss_start} years'],
        ['SS Monthly Benefit (at FRA):', f'{currency_symbol}{base_ss:,.0f}/month']
    ]
    
    assumptions_table = Table(assumptions_data, colWidths=[3*inch, 2*inch])
    assumptions_style = TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightyellow),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ])
    assumptions_table.setStyle(assumptions_style)
    story.append(assumptions_table)
    story.append(Spacer(1, 20))
    
    # Wealth Summary (if applicable)
    if savings_accounts or planned_expenses:
        story.append(Paragraph("Wealth Accumulation Summary", heading_style))
        
        enabled_accounts = [acc for acc in savings_accounts if acc.get('enabled', True)]
        enabled_expenses = [exp for exp in planned_expenses if exp.get('enabled', True)]
        
        if enabled_accounts:
            if DEBUG_LOGGING:
                logger.info(f"📋 Processing {len(enabled_accounts)} enabled accounts for PDF")
                for i, account in enumerate(enabled_accounts):
                    logger.info(f"  Account {i}: name='{account.get('name', 'MISSING')}', amount={account.get('amount', 0)}")
            
            hebrew_heading3_style = ParagraphStyle(
                'CustomHeading3',
                parent=styles['Heading3'],
                fontName=hebrew_font_name,
                fontSize=12
            )
            story.append(Paragraph("Savings Accounts:", hebrew_heading3_style))
            account_data = [['Account Name', 'Current Amount', 'ROI', 'Monthly Deposit', 'Projected Savings']]
            for account in enabled_accounts:
                account_name = account['name']
                if DEBUG_LOGGING:
                    logger.info(f"  📝 Adding account to PDF: '{account_name}' (type: {type(account_name)})")
                    logger.info(f"      Raw bytes: {account_name.encode('utf-8')}")
                    logger.info(f"      Contains Hebrew: {contains_hebrew(account_name)}")
                
                # Process Hebrew text for proper RTL display
                processed_account_name = process_hebrew_text(account_name)
                
                # Calculate projected value at retirement
                years_to_retirement = ret_age - user_age
                projected_value = 0
                
                # Calculate growth from initial amount
                if account['amount'] > 0:
                    projected_value += account['amount'] * (1 + account['roi']) ** years_to_retirement
                
                # Calculate growth from monthly deposits (Future Value of Annuity)
                monthly_deposit = account.get('monthly_deposit', 0)
                if monthly_deposit > 0 and years_to_retirement > 0:
                    months_to_retirement = years_to_retirement * 12
                    monthly_rate = account['roi'] / 12
                    if monthly_rate > 0:
                        # FV of ordinary annuity formula
                        annuity_value = monthly_deposit * ((1 + monthly_rate) ** months_to_retirement - 1) / monthly_rate
                    else:
                        # If no interest, just sum the deposits
                        annuity_value = monthly_deposit * months_to_retirement
                    projected_value += annuity_value
                
                account_data.append([
                    processed_account_name,
                    f'{currency_symbol}{account["amount"]:,.0f}',
                    f'{account["roi"]:.1%}',
                    f'{currency_symbol}{account.get("monthly_deposit", 0):,.0f}/mo',
                    f'{currency_symbol}{projected_value:,.0f}'
                ])
            
            accounts_table = Table(account_data, colWidths=[1.2*inch, 1.0*inch, 0.6*inch, 1.0*inch, 1.2*inch])
            accounts_style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),  # Headers use Helvetica
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]
            
            # Apply Hebrew font only to cells with Hebrew content
            for row_idx in range(1, len(account_data)):  # Skip header row
                for col_idx in range(len(account_data[row_idx])):
                    cell_content = account_data[row_idx][col_idx]
                    if contains_hebrew(str(cell_content)):
                        accounts_style.append(('FONTNAME', (col_idx, row_idx), (col_idx, row_idx), hebrew_font_name))
                    else:
                        accounts_style.append(('FONTNAME', (col_idx, row_idx), (col_idx, row_idx), 'Helvetica'))
            
            table_style = TableStyle(accounts_style)
            accounts_table.setStyle(table_style)
            story.append(accounts_table)
            story.append(Spacer(1, 12))
        
        if enabled_expenses:
            story.append(Paragraph("Planned Expenses:", hebrew_heading3_style))
            expense_data = [['Expense Name', 'Amount', 'Age']]
            for expense in enabled_expenses:
                # Process Hebrew text for proper RTL display
                processed_expense_name = process_hebrew_text(expense['name'])
                
                expense_data.append([
                    processed_expense_name,
                    f'{currency_symbol}{expense["amount"]:,.0f}',
                    f'{expense["age"]} years'
                ])
            
            expenses_table = Table(expense_data, colWidths=[2*inch, 1.5*inch, 1.3*inch])
            expenses_style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica'),  # Headers use Helvetica
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]
            
            # Apply Hebrew font only to cells with Hebrew content
            for row_idx in range(1, len(expense_data)):  # Skip header row
                for col_idx in range(len(expense_data[row_idx])):
                    cell_content = expense_data[row_idx][col_idx]
                    if contains_hebrew(str(cell_content)):
                        expenses_style.append(('FONTNAME', (col_idx, row_idx), (col_idx, row_idx), hebrew_font_name))
                    else:
                        expenses_style.append(('FONTNAME', (col_idx, row_idx), (col_idx, row_idx), 'Helvetica'))
            
            table_style = TableStyle(expenses_style)
            expenses_table.setStyle(table_style)
            story.append(expenses_table)
            story.append(Spacer(1, 20))
    
    # Add chart on new page
    from reportlab.platypus import PageBreak
    story.append(PageBreak())
    story.append(Paragraph("Annual Financial Projection", heading_style))
    
    # Generate and embed chart
    chart_buffer = export_chart_for_pdf(df, config['symbol'], wealth_df)
    chart_img = Image(chart_buffer, width=6*inch, height=3.6*inch)
    story.append(chart_img)
    story.append(Spacer(1, 20))
    
    # Data table
    story.append(Paragraph("Annual Spending & Income Detail (First 20 Years)", heading_style))
    
    # Prepare table data
    table_data = [display_df.columns.tolist()]
    for _, row in display_df.iterrows():
        table_data.append(row.tolist())
    
    data_table = Table(table_data, colWidths=[0.6*inch, 1.1*inch, 1.1*inch, 1.0*inch, 1.1*inch])
    data_style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), hebrew_font_name),
        ('FONTNAME', (0, 1), (-1, -1), hebrew_font_name),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey])
    ])
    data_table.setStyle(data_style)
    story.append(data_table)
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    
    return buffer.getvalue()


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
            # Skip disabled accounts
            if not account.get('enabled', True):
                continue
                
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
        # Skip disabled expenses
        if not expense.get('enabled', True):
            continue
            
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


def calculate_target_monthly_spend(target_nest_egg: float, spouse: bool, ret_age: int, user_age: int, 
                                 ss_start: int, base_ss: float, config: dict, 
                                 inflation_rate: float, discount_rate: float,
                                 early_decline: float, couple_decline: float, single_decline: float) -> float:
    """Calculate the monthly spend that results in a specific nest egg target using binary search."""
    
    def calculate_nest_egg_for_monthly_spend(monthly_spend: float) -> float:
        """Helper function to calculate nest egg for a given monthly spend."""
        # Apply inflation adjustment
        years_to_retirement = ret_age - user_age
        inflated_monthly_spend = monthly_spend * (1 + inflation_rate) ** years_to_retirement
        
        # Calculate spending projections
        df = project_spending(inflated_monthly_spend, spouse, ret_age, config['symbol'], 
                            early_decline, couple_decline, single_decline)
        df = add_social_security(df, ss_start, base_ss, spouse, config)
        
        # Calculate present value of needs
        return pv_of_needs(df, ret_age, discount_rate)
    
    # Binary search bounds - reasonable monthly spending range
    low_spend = 500.0   # $500/month minimum
    high_spend = 200000.0  # $200,000/month maximum (matches validation limit)
    tolerance = 100.0   # Within $100 of target
    max_iterations = 50
    
    # Binary search for the target monthly spend
    for _ in range(max_iterations):
        mid_spend = (low_spend + high_spend) / 2
        calculated_nest_egg = calculate_nest_egg_for_monthly_spend(mid_spend)
        
        # Check if we're close enough to the target
        if abs(calculated_nest_egg - target_nest_egg) <= tolerance:
            return mid_spend
        
        # Adjust search bounds
        if calculated_nest_egg < target_nest_egg:
            low_spend = mid_spend
        else:
            high_spend = mid_spend
    
    # Return the closest result if we didn't converge
    return (low_spend + high_spend) / 2


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
    
    # Initialize session state for scenario tracking
    if 'selected_scenario' not in st.session_state:
        st.session_state.selected_scenario = "Moderate"

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
        col1, col2 = st.columns([4, 1])
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
                        'monthly_deposit': 0,
                        'enabled': True
                    })
                    st.rerun()
        
        # Display existing savings accounts
        accounts_to_remove = []
        for i, account in enumerate(st.session_state.savings_accounts):
            with st.container():
                # Row 1: Account name, enable/disable toggle, and remove button
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.text(f"🏦 {account['name']}")
                
                with col2:
                    # Enable/disable toggle
                    toggle_text = "✅" if account.get('enabled', True) else "❌"
                    if st.button(toggle_text, key=f"toggle_account_{i}", help="Enable/disable account"):
                        account['enabled'] = not account.get('enabled', True)
                        st.rerun()
                
                with col3:
                    if st.button("🗑️", key=f"remove_account_{i}", help="Remove account"):
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
        col1, col2 = st.columns([4, 1])
        with col1:
            new_expense_name = st.text_input("Expense name", placeholder="e.g., Wedding, House, Car", key="new_expense_name")
        with col2:
            st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)  # Align button with input bottom
            if st.button("➕", help="Add expense"):
                if new_expense_name:
                    st.session_state.planned_expenses.append({
                        'name': new_expense_name,
                        'amount': 0,
                        'age': user_age + 5,
                        'enabled': True
                    })
                    st.rerun()
        
        # Display existing planned expenses
        expenses_to_remove = []
        for i, expense in enumerate(st.session_state.planned_expenses):
            with st.container():
                # Row 1: Expense name, enable/disable toggle, and remove button
                col1, col2, col3 = st.columns([4, 1, 1])
                
                with col1:
                    st.text(f"💸 {expense['name']}")
                
                with col2:
                    # Enable/disable toggle
                    toggle_text = "✅" if expense.get('enabled', True) else "❌"
                    if st.button(toggle_text, key=f"toggle_expense_{i}", help="Enable/disable expense"):
                        expense['enabled'] = not expense.get('enabled', True)
                        st.rerun()
                
                with col3:
                    if st.button("🗑️", key=f"remove_expense_{i}", help="Remove expense"):
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
            value=float(st.session_state.monthly_spend or config["default_monthly_spend"]), 
            step=100.0,
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
            index=list(SCENARIO_PRESETS.keys()).index(st.session_state.selected_scenario),
            help="Choose conservative, moderate, or optimistic assumptions"
        )
        
        # Detect scenario change
        if scenario != st.session_state.selected_scenario:
            # Scenario changed - update all assumption values in session state
            scenario_changed = True
            st.session_state.selected_scenario = scenario
        else:
            scenario_changed = False
        
        preset = SCENARIO_PRESETS[scenario]
        st.info(preset["description"])
        
        # Update session state values when scenario changes
        if scenario_changed:
            # Update all assumption values to match new preset
            st.session_state.discount_rate = preset["discount_rate"]
            st.session_state.early_decline = preset["early_decline"]
            st.session_state.couple_decline = preset["couple_decline"]
            st.session_state.single_decline = preset["single_decline"]
            
            # Show notification to user
            st.success(f"✅ Updated assumptions for {scenario} scenario")
        
        # Use session state values (updated by scenario change or user customization)
        discount_rate = st.session_state.discount_rate or preset["discount_rate"]
        early_decline = st.session_state.early_decline or preset["early_decline"]
        couple_decline = st.session_state.couple_decline or preset["couple_decline"]
        single_decline = st.session_state.single_decline or preset["single_decline"]
        
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
                label=f"**Total expected savings**", 
                value=format_currency(wealth_at_retirement, currency_symbol),
                help="Total amount your current savings and planned expenses will be worth at retirement"
            )
            
            st.metric(
                label=f"**Total retirement need**", 
                value=format_currency(retirement_analysis['total_required'], currency_symbol),
                help="Total nest egg required before considering existing wealth"
            )
            
            # Add "Adjust to Projected Savings" button when wealth contribution exists
            if retirement_analysis['wealth_contribution'] > 0:
                if st.button(
                    "🎯 Adjust to Projected Savings", 
                    help="Automatically adjust monthly spending so your total retirement need matches your projected savings",
                    key="adjust_to_savings_btn"
                ):
                    try:
                        # Calculate target monthly spend
                        target_monthly_spend = calculate_target_monthly_spend(
                            target_nest_egg=wealth_at_retirement,
                            spouse=spouse,
                            ret_age=ret_age,
                            user_age=user_age,
                            ss_start=ss_start,
                            base_ss=base_ss,
                            config=config,
                            inflation_rate=inflation_rate,
                            discount_rate=discount_rate,
                            early_decline=early_decline,
                            couple_decline=couple_decline,
                            single_decline=single_decline
                        )
                        
                        # Validate the result is reasonable
                        if target_monthly_spend < 100 or target_monthly_spend > 200000:
                            st.error(
                                f"❌ **Unable to adjust spending**\n\n"
                                f"The calculated spending ({format_currency(target_monthly_spend, currency_symbol)}/month) "
                                f"is outside reasonable bounds. Try adjusting your savings accounts or assumptions."
                            )
                        else:
                            # Store old value for feedback
                            old_monthly_spend = st.session_state.monthly_spend
                            
                            # Update session state with new monthly spend
                            st.session_state.monthly_spend = target_monthly_spend
                            
                            # Show success message
                            st.success(
                                f"✅ **Adjusted monthly spending**\n\n"
                                f"**From:** {format_currency(old_monthly_spend, currency_symbol)}/month\n\n"
                                f"**To:** {format_currency(target_monthly_spend, currency_symbol)}/month\n\n"
                                f"Your retirement needs now match your projected savings!"
                            )
                            
                            # Rerun to refresh calculations
                            st.rerun()
                    
                    except Exception as e:
                        st.error(
                            f"❌ **Calculation error**\n\n"
                            f"Unable to calculate adjusted spending. Please check your inputs and try again."
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
        
        # Export to PDF button
        st.subheader("📄 Export Report")
        if st.button(
            "📄 Export to PDF",
            help="Download a comprehensive PDF report with all assumptions, calculations, chart, and data table",
            key="export_pdf_btn",
            use_container_width=True
        ):
            try:
                with st.spinner("Generating PDF report..."):
                    # Determine the selected scenario
                    current_scenario = st.session_state.get('selected_scenario', 'Moderate')
                    
                    # Create display_df for PDF (same as used in the main table)
                    display_df_for_pdf = df[["Age", "Gross_Spending", "Social_Security", "Net_Need", "Savings_Balance"]].head(20).copy()
                    
                    # Format currency columns
                    for col in ["Gross_Spending", "Social_Security", "Net_Need", "Savings_Balance"]:
                        display_df_for_pdf[col] = display_df_for_pdf[col].apply(lambda x: format_currency(x, currency_symbol))
                    
                    # Rename columns for display
                    display_df_for_pdf.columns = ["Age", f"Gross Spending ({config['currency']})", 
                                                 f"{ss_name} ({config['currency']})", f"Net Need ({config['currency']})",
                                                 f"Savings Balance ({config['currency']})"]
                    
                    # Generate PDF
                    pdf_bytes = generate_pdf_report(
                        retirement_analysis=retirement_analysis,
                        wealth_at_retirement=wealth_at_retirement,
                        nest_egg=nest_egg,
                        df=df,
                        display_df=display_df_for_pdf,
                        config=config,
                        user_age=user_age,
                        ret_age=ret_age,
                        spouse=spouse,
                        monthly_spend=monthly_spend,
                        ss_start=ss_start,
                        base_ss=base_ss,
                        discount_rate=discount_rate,
                        inflation_rate=inflation_rate,
                        early_decline=early_decline,
                        couple_decline=couple_decline,
                        single_decline=single_decline,
                        years_to_retirement=years_to_retirement,
                        savings_accounts=st.session_state.savings_accounts,
                        planned_expenses=st.session_state.planned_expenses,
                        scenario=current_scenario,
                        wealth_df=wealth_df
                    )
                    
                    # Create download button
                    filename = f"retirement_analysis_{config['currency']}_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                    st.download_button(
                        label="💾 Download PDF Report",
                        data=pdf_bytes,
                        file_name=filename,
                        mime="application/pdf",
                        use_container_width=True
                    )
                    
                    st.success("✅ PDF report generated successfully! Click the download button above.")
                    
            except Exception as e:
                st.error(
                    f"❌ **PDF Generation Error**\n\n"
                    f"Unable to generate PDF report. Please try again or contact support.\n\n"
                    f"Error details: {str(e)}"
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
            current_total = sum(account['amount'] for account in st.session_state.savings_accounts if account.get('enabled', True))
            total_expenses = sum(expense['amount'] for expense in st.session_state.planned_expenses if expense.get('enabled', True))
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
