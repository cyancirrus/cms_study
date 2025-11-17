#!/bin/bash
set -e
# sqlfluff lint --dialect sqlite etl
sqlfluff fix --dialect sqlite etl

# sqlformat --reindent --keywords upper -i .
# # # Loop over staged files passed by Lefthook
# # for f in "$@"; do
# #     sqlformat "$f" --reindent --keywords upper -i
# # done

# # git add "$@"

# # # git add --all :/
