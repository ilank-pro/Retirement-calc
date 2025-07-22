# 💰 Multi-Currency Retirement Needs Calculator

A comprehensive retirement planning tool that estimates the nest-egg required to retire comfortably across different countries and currencies, using empirically observed spending decline patterns and country-specific social security systems.

## 🌍 Supported Countries & Currencies

- **🇺🇸 United States (USD)** - Social Security
- **🇪🇺 European Union (EUR)** - Average European pension systems
- **🇬🇧 United Kingdom (GBP)** - State Pension
- **🇮🇱 Israel (NIS)** - Old Age Pension (Bituach Leumi)

## ✨ Features

### Multi-Currency Support
- Automatic currency formatting with proper symbols
- Country-specific default values and benefit calculations
- Real-time currency switching

### Country-Specific Social Security Systems
- **USA**: Full retirement age 67, early claiming from 62, delayed retirement credits until 70
- **Europe**: Average EU system with €1,345/month average pension
- **UK**: State Pension age 66, £962/month full pension, no early claiming
- **Israel**: Retirement age 67, ₪1,795/month basic pension with spousal increments

### Advanced Retirement Planning
- Empirically-based spending decline patterns (RAND 2023, CRR 2021)
- Present value calculations with 7% real return assumption
- Spouse benefit calculations
- Interactive projections and visualizations

### User-Friendly Interface
- Interactive Streamlit web application
- Real-time benefit multiplier calculations
- Detailed annual breakdowns
- Country-specific information panels

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ilank-pro/Retirement-calc.git
   cd Retirement-calc
   ```

2. **Create and activate virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run retirement_calculator.py
   ```

5. **Open your browser**
   - The app will automatically open at `http://localhost:8501`

## 📊 How It Works

### Spending Decline Model
The calculator uses research-backed spending patterns:
- **Ages 62-64**: 1% annual decline (transition from "go-go" to "slow-go" years)
- **Ages 65+**: 
  - Couples: 2.4% annual decline
  - Singles: 1.7% annual decline

### Social Security Calculations
Each country has specific rules:
- **Benefit multipliers** based on claiming age
- **Spousal benefits** where applicable
- **Early/late claiming penalties and bonuses**

### Present Value Analysis
- 7% real return assumption during retirement
- Projects expenses and income through age 95
- Calculates required nest-egg at retirement

## 💡 Usage Tips

1. **Select Your Country**: Choose your country/region to get accurate social security rules and default values
2. **Input Your Details**: Enter your expected monthly spending and estimated social security benefit
3. **Adjust Timing**: Experiment with different retirement and social security claiming ages
4. **Review Results**: Check the required nest-egg, benefit coverage, and annual projections

## 📈 Key Outputs

- **Required Nest-Egg**: Present value of retirement needs minus social security
- **Social Security Coverage**: Percentage of expenses covered by government benefits
- **Annual Projections**: Year-by-year breakdown of spending, benefits, and net needs
- **Interactive Charts**: Visual representation of your retirement financial flow

## 🔢 Data Sources

All calculations use official government data:
- **USA**: Social Security Administration (SSA) 2024 data
- **Europe**: Eurostat 2022, average EU pension expenditure
- **UK**: Gov.UK 2025-26 State Pension rates
- **Israel**: National Insurance Institute (Bituach Leumi) 2025 rates

## 🛠️ Technical Details

### Built With
- **Streamlit** - Web application framework
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computations
- **Matplotlib** - Data visualization

### Key Assumptions
- Real investment returns: 7% during retirement
- Inflation-adjusted social security benefits
- Consistent spending patterns across countries
- Life expectancy: 95 years

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Clone your fork
git clone https://github.com/your-username/Retirement-calc.git
cd Retirement-calc

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run in development mode
streamlit run retirement_calculator.py
```

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 📞 Support

If you have questions or need help:
1. Check the in-app help tooltips
2. Review the methodology section in the app
3. Open an issue on GitHub
4. Consult the official government sources for social security information

## 🎯 Future Enhancements

- [ ] Additional countries and currencies
- [ ] Tax considerations
- [ ] Healthcare cost projections
- [ ] Monte Carlo simulations
- [ ] Export functionality
- [ ] Mobile optimization

---

**Disclaimer**: This calculator provides estimates for educational purposes. Consult with a financial advisor for personalized retirement planning advice. Social security rules and benefits may change over time. 