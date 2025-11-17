#!/bin/bash
set -e
# # NOTE: Uncomment lint this if lost why failing
# sqlfluff lint --dialect sqlite etl

sqlfluff fix --dialect sqlite etl

git add --all :/
