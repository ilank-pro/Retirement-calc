#!/bin/bash

# Multi-Currency Retirement Calculator Startup Script
# This script sets up the environment and runs the Streamlit application

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BLUE}🏦 Multi-Currency Retirement Calculator${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Python3 found: $(python3 --version)${NC}"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}📦 Virtual environment not found. Creating one...${NC}"
    python3 -m venv venv
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Virtual environment created successfully${NC}"
    else
        echo -e "${RED}❌ Failed to create virtual environment${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Virtual environment found${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source venv/bin/activate

# Check if requirements are installed
if [ ! -f "venv/lib/python*/site-packages/streamlit/__init__.py" ] && [ ! -f "venv/lib/python*/site-packages/streamlit/__init__.py" ]; then
    echo -e "${YELLOW}📥 Installing dependencies...${NC}"
    pip install -r requirements.txt
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Dependencies installed successfully${NC}"
    else
        echo -e "${RED}❌ Failed to install dependencies${NC}"
        exit 1
    fi
else
    echo -e "${GREEN}✅ Dependencies already installed${NC}"
fi

# Check if the main application file exists
if [ ! -f "retirement_calculator.py" ]; then
    echo -e "${RED}❌ retirement_calculator.py not found in current directory${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Application file found${NC}"
echo ""

# Display startup information
echo -e "${BLUE}🚀 Starting Multi-Currency Retirement Calculator...${NC}"
echo -e "${YELLOW}📍 The application will open in your default web browser${NC}"
echo -e "${YELLOW}🔗 URL: http://localhost:8501${NC}"
echo ""
echo -e "${BLUE}💡 Supported Countries:${NC}"
echo -e "   🇺🇸 United States (USD) - Social Security"
echo -e "   🇪🇺 European Union (EUR) - Average EU pension"
echo -e "   🇬🇧 United Kingdom (GBP) - State Pension"
echo -e "   🇮🇱 Israel (NIS) - Old Age Pension"
echo ""
echo -e "${YELLOW}⏹️  To stop the application, press Ctrl+C${NC}"
echo ""

# Function to handle cleanup on exit
cleanup() {
    echo ""
    echo -e "${BLUE}🛑 Shutting down application...${NC}"
    echo -e "${GREEN}✅ Thank you for using the Multi-Currency Retirement Calculator!${NC}"
    deactivate 2>/dev/null || true
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Run the Streamlit application
streamlit run retirement_calculator.py

# If we get here, streamlit exited normally
cleanup 