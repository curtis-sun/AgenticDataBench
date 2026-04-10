#!/bin/bash

cd "$(dirname "$0")/da_agent/images/da-agent"
docker build -t da-agent .

cd "$(dirname "$0")"
python3 run.py --agent da-agent --suffix qwen --model qwen3.5-397b-a17b --port 30008 --max_steps 50