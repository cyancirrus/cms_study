#!/bin/bash
set -e # exit immediately if any command fails

shfmt -w ./scripts
git add --all :/
