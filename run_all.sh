#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/experiments"
python3 exp_stability.py
python3 exp_stability.py extra
python3 exp_european.py
python3 exp_american.py
python3 exp_greeks.py
python3 exp_hurst.py
python3 exp_efficiency.py   # run alone for meaningful timings
python3 make_tables.py
python3 make_figures.py
