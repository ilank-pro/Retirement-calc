# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Quick Start
```bash
./start.sh
```
The startup script handles environment setup, dependency installation, and launches the Streamlit app.

### Manual Development Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run retirement_calculator.py
```

### Dependencies
Core dependencies are listed in `requirements.txt`:
- `streamlit` - Web application framework
- `numpy` - Numerical computations
- `pandas` - Data manipulation and analysis
- `matplotlib` - Data visualization

## Architecture Overview

This is a single-file Streamlit application (`retirement_calculator.py`) that implements a multi-currency retirement calculator with country-specific social security systems.

### Core Components

**Country Configuration System (`COUNTRY_CONFIG`)**
- Centralized configuration for USA, Europe, UK, and Israel
- Each country defines: currency, social security rules, default values, benefit multipliers
- Located at `retirement_calculator.py:42-95`

**Social Security Calculations (`ss_multiplier` function)**
- Calculates benefit multipliers based on claiming age vs. full retirement age
- Handles early claiming penalties and delayed retirement credits
- Country-specific rules for early/late claiming limits
- Located at `retirement_calculator.py:98-114`

**Spending Projection Model (`project_spending` function)**
- Implements empirical spending decline patterns from research (RAND 2023, CRR 2021)
- Age-based spending phases: 1% decline ages 62-64, then 2.4% (couples) or 1.7% (singles) decline
- Projects annual spending from retirement age through age 95
- Located at `retirement_calculator.py:117-129`

**Financial Modeling (`add_social_security`, `pv_of_needs` functions)**
- Integrates social security benefits with spending projections
- Calculates present value using 7% real return assumption
- Determines net retirement funding needs after social security
- Located at `retirement_calculator.py:132-145`

**Streamlit UI (`main` function)**
- Sidebar for country selection and input parameters
- Real-time calculations and visualizations
- Country-specific defaults and information panels
- Multi-column layout with metrics and charts
- Located at `retirement_calculator.py:153-330`

### Key Constants
- `MAX_AGE = 95` - Life expectancy assumption
- `POST_RET_ROI = 0.07` - Real return assumption during retirement
- Spending decline rates: `EARLY_DECLINE = 0.01`, `COUPLE_DECLINE = 0.024`, `SINGLE_DECLINE = 0.017`

### Data Flow
1. User selects country → loads country-specific configuration
2. User inputs financial parameters → uses country defaults as starting values  
3. Spending projection calculated using empirical decline patterns
4. Social security benefits calculated with country-specific multipliers
5. Present value analysis determines required nest-egg
6. Results displayed with interactive charts and detailed breakdowns

### Country-Specific Features
- **USA**: Social Security with 67 FRA, early claiming from 62, delayed credits to 70
- **Europe**: Average EU pension system with 65 FRA, variable early/late claiming
- **UK**: State Pension with 66 FRA, no early claiming, deferral bonuses
- **Israel**: Old Age Pension with 67 FRA, early claiming from 62, spousal increments

## File Structure
```
retirement_calc/
├── retirement_calculator.py  # Main application (single file)
├── requirements.txt         # Python dependencies
├── start.sh                # Startup script with environment setup
├── README.md               # Comprehensive documentation
├── .gitignore             # Standard Python gitignore
└── venv/                  # Virtual environment (not in git)
```

## Development Notes

- This is a single-file application - all logic is contained in `retirement_calculator.py`
- No testing framework is currently implemented
- No build process required - direct Python execution via Streamlit
- Application runs on `http://localhost:8501` by default
- Uses Streamlit's built-in hot reloading for development