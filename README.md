# Learning Curve Analyzer

**Learning curves** for bias-variance diagnosis — train/validation score vs dataset size.

## Overview

- Train-validation score curves with std bands
- Train-val gap metric for overfitting detection
- Interactive Plotly visualization
- **Streamlit dashboard**

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
# CLI: python -m src.main analyze --dataset wine
pytest tests/ -v
```

## Docker

```bash
docker compose up --build
```

## License

MIT
# Learning-Curve-Analyzer
