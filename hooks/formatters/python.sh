#!/bin/bash
set -e # exit immediately if any command fails

black --line-length 72 .
git add --all :/
