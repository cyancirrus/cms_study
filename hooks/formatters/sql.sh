#!/bin/bash
set -e

# Loop over staged files passed by Lefthook
for f in "$@"; do
    sqlformat "$f" --reindent --keywords upper -i
done

git add "$@"

# git add --all :/
