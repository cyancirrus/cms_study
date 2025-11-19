#!/bin/bash
set -e # exit immediately if any command fails

./scripts/model_performance.sh
git add --all :/
