#!/bin/bash
set -e # exit immediately if any command fails

sqlformat . --reindent --keywords upper -i
# git add --all :/
