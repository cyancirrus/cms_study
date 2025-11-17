#!/bin/bash
set -e # exit immediately if any command fails

PYTHONPATH=./
pytest ./tests
