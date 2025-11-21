# OFI Analysis Dashboards

## Available Dashboards

### 1. Research Dashboard (`dashboard_research.py`) - NEW! 🎉
Complete Cont et al. (2011) replication dashboard with all analyses

**Features:**
- 📈 Overview with all summary tables
- 📊 Regression Analysis (Table 2, Figure 2)
- 📉 Depth Analysis (Table 3, Figure 4)
- 🎯 Event Patterns (Table 4, Figure 5)
- ⚖️ OFI vs TI (Table 5, Figure 6)
- 🔬 Multi-Market Comparison
- 🖼️ Figures Gallery (all PNG + PDF)

**Run:** `streamlit run dashboard_research.py`

### 2. Data Explorer (`dashboard_simple.py`)
Fast interactive exploration of raw OFI data

**Run:** `streamlit run dashboard_simple.py`

## Quick Start

```bash
# Generate all results
python scripts/run_all_analyses.py

# Launch research dashboard
streamlit run dashboard_research.py
```

## Documentation

See REPLICATION_SUMMARY.md for complete research findings.
