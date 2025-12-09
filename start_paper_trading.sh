#!/bin/bash
cd /home/julian/dev/hobby/strategy-simulator
source .venv/bin/activate
python tasks/cli_trader.py --strategy buy-the-dip --mode paper "$@"
