#!/usr/bin/env bash
set -euo pipefail

#TODO: Refactor if going to use this script at all

# Usage:
#   ./move_keep_files.sh /path/to/target_dir
#
# Example:
#   ./move_keep_files.sh ../clean_2024
#
# Run this from inside the folder that has all the CSVs.

target_dir="${1:-}"

if [[ -z "$target_dir" ]]; then
	echo "Usage: $0 /path/to/target_dir"
	exit 1
fi

mkdir -p "$target_dir"

keep_patterns=(
	"Complications_and_Deaths-Hospital.csv"
	"FY_202*_HAC_Reduction_Program_Hospital.csv"
	"FY_202*_Hospital_Readmissions_Reduction_Program_Hospital.csv"
	"HCAHPS-Hospital.csv"
	"Hospital_General_Information.csv"
	"IPFQR_QualityMeasures_Facility.csv"
	"Medicare_Hospital_Spending_Per_Patient-Hospital.csv"
	"PCH_HCAHPS_HOSPITAL.csv"
	"Timely_and_Effective_Care-Hospital.csv"
	"Unplanned_Hospital_Visits-Hospital.csv"
	"hvbp_clinical_outcomes.csv"
	"hvbp_efficiency_and_cost_reduction.csv"
	"hvbp_person_and_community_engagement.csv"
	"hvbp_safety.csv"
	"hvbp_tps.csv"
)

echo "Current directory: $PWD"
echo "Target directory:  $target_dir"
echo
echo "Will MOVE files matching these patterns:"
for p in "${keep_patterns[@]}"; do
	echo "  - $p"
done
echo

read -p "Type 'yes' to continue: " ans
if [[ "$ans" != "yes" ]]; then
	echo "Aborted."
	exit 1
fi

shopt -s nullglob

for pattern in "${keep_patterns[@]}"; do
	matches=($pattern)
	for f in "${matches[@]}"; do
		if [[ -f "$f" ]]; then
			echo "Moving: $f -> $target_dir/"
			mv -n -- "$f" "$target_dir/"
			# -n so it won't overwrite if you accidentally run twice
		fi
	done
done

shopt -u nullglob

echo "Done moving matching files."
echo "You can inspect '$target_dir' and then manually clean up the rest if you want."
