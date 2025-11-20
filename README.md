# Hospital Quality Analytics

A predictive analytics system for Medicare hospital performance, focusing on mortality rates, complications, patient safety, and quality of care outcomes.

## Overview

This project analyzes CMS (Center for Medicare and Medicaid Services) hospital data to predict year-over-year changes in hospital performance metrics. The system combines hospital outcomes data with demographic and geographic context to generate actionable recommendations.

## Data Notes

- The `data/` directory contains cleaned datasets for multiple years and is **not included in the repository** due to size constraints.
- The SQLite database (`*.db`) is also excluded for the same reason.
- If you wish to run the full pipeline with all years of data, please contact the author or consider using Git LFS to manage large files.
- All scripts are written to work with the data directory structure as shown; placeholder or smaller test data can be used to validate functionality.

## Quick Start

```bash
# Enable virtual environment
source ../bin/activate

# Run ETL pipeline
./scripts/etl_execute.sh

# Launch API server
./scripts/launch_api.sh

# Test API endpoints
./scripts/requests.sh

# Stop API server
./scripts/terminate_api.sh

# Generate visualizations
./scripts/visualization_histograms.sh

# Check model performance
./scripts/model_performance.sh
```

## Key Definitions

| Acronym | Definition |
|---------|------------|
| **CMS** | Center for Medicare and Medicaid Services |
| **HVBP** | Hospital Value-Based Purchasing |
| **HAC** | Hospital-Acquired Condition Reduction Program |
| **HCAHPS** | Hospital Consumer Assessment of Healthcare Providers and Systems |
| **IPFQR** | Inpatient Psychiatric Facility Quality Reporting Program |
| **Medicare** | Federal health insurance for elderly (65+) and certain disabled populations |
| **Medicaid** | State + federal program for low-income individuals |
| **MSA** | Metropolitan Statistical Area |

## Data Sources

### Primary CMS Data

Current year data is fetched via CMS metadata API:
```bash
curl "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/avtz-f2ge"
```

Historical data (2021-2024) from archived datasets:
- https://data.cms.gov/provider-data/archived-data/hospitals

Data dictionary reference:
- https://data.cms.gov/provider-data/sites/default/files/data_dictionaries/hospital/HOSPITAL_Data_Dictionary.pdf

### Geographic & Demographic Data

**ZIP Code Data**: https://download.geonames.org/export/zip/ (US dataset)

**State Regions** (Census 2017): https://www2.census.gov/programs-surveys/bps/guidance/states-by-region.pdf

**MSA Coordinates**: https://data-usdot.opendata.arcgis.com/datasets/usdot::core-based-statistical-areas/about

**MSA Income Statistics**: https://www.bea.gov/data/income-saving/personal-income-county-metro-and-other-areas

**MSA Definitions**: 
- https://www.census.gov/geographies/reference-maps/2020/geo/cbsa.html
- https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2015/delineation-files/list1.xlsx

## Dataset Descriptions

### Hospital Outcomes & Quality
- `Complications_and_Deaths-Hospital.csv` - Mortality and complication rates for various conditions
- `Timely_and_Effective_Care-Hospital.csv` - Process measure compliance
- `Unplanned_Hospital_Visits-Hospital.csv` - Readmission metrics

### Payment & Penalty Programs
- `FY_2025_HAC_Reduction_Program_Hospital.csv` - Safety penalties
- `FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv` - Readmission penalties
- `hvbp_clinical_outcomes.csv` - Value-based purchasing: clinical domain
- `hvbp_efficiency_and_cost_reduction.csv` - Value-based purchasing: cost domain
- `hvbp_person_and_community_engagement.csv` - Value-based purchasing: engagement domain
- `hvbp_safety.csv` - Value-based purchasing: safety domain
- `hvbp_tps.csv` - Total performance scores

### Patient Experience
- `HCAHPS-Hospital.csv` - General hospital patient satisfaction
- `PCH_HCAHPS_HOSPITAL.csv` - Pediatric hospital patient satisfaction

### Financial & Specialty
- `Medicare_Hospital_Spending_Per_Patient-Hospital.csv` - Cost per episode
- `IPFQR_QualityMeasures_Facility.csv` - Psychiatric facility quality measures
- `Hospital_General_Information.csv` - Facility characteristics and metadata

### Geographic Reference
- `RUCA-codes-2020-tract.csv` - Rural-Urban Commuting Area codes (tract level)
- `RUCA-codes-2020-zipcode.csv` - RUCA codes (ZIP level)

## Fiscal Year Definition

**FY2025** = October 1, 2024 → September 30, 2025

- October 1 is inclusive, September 30 is inclusive
- A measure dated Sep 30, 2025 still belongs to FY25
- Any timestamp after Sep 30, 2025 23:59:59 rolls into FY26

## Data Normalization Notes

### Complications and Deaths Measures

**Per 100 patients:**
- `COMP_HIP_KNEE` - Rate of complications for hip/knee replacement
- `MORT_30_AMI` - 30-day death rate for heart attack
- `MORT_30_CABG` - 30-day death rate for CABG surgery
- `MORT_30_COPD` - 30-day death rate for COPD
- `MORT_30_HF` - 30-day death rate for heart failure
- `MORT_30_PN` - 30-day death rate for pneumonia
- `MORT_30_STK` - 30-day death rate for stroke
- `Hybrid_HWM` - Hospital-Wide All-Cause Risk Standardized Mortality (ratio × 100)

**Per 1000 patients (PSI measures):**
- `PSI_03` - Pressure ulcer rate
- `PSI_04` - Death rate among surgical inpatients with serious treatable complications
- `PSI_06` - Iatrogenic pneumothorax rate
- `PSI_08` - In-hospital fall-associated fracture rate
- `PSI_09` - Postoperative hemorrhage or hematoma rate
- `PSI_10` - Postoperative acute kidney injury requiring dialysis rate
- `PSI_11` - Postoperative respiratory failure rate
- `PSI_12` - Perioperative pulmonary embolism or DVT rate
- `PSI_13` - Postoperative sepsis rate
- `PSI_14` - Postoperative wound dehiscence rate
- `PSI_15` - Abdominopelvic accidental puncture or laceration rate
- `PSI_90` - CMS Medicare PSI 90: Patient safety and adverse events composite

⚠️ **Note**: PSI_04 has extremely different scale (mean ~175 per 1000) - divide by 1000 for consistency.

## Feature Engineering

### MSA-Based Demographics

**Population Influence Model:**
```
Population_influence = Σᵢ Population_i × exp(-distance_i² / (2 × σ²))
```
- Decay function from nearest MSA centroid
- Optional: Sum top k nearest MSAs weighted by decay
- Captures urban/rural gradient

**Income Proxy:**
- Uses nearest MSA's median or per-capita income
- Static socio-economic indicator

**Rationale:**
- Provides demographic context beyond state/city/county
- Avoids sparse data at granular geographic levels
- **Population density** indicates: urban vs rural, specialist availability, competition
- **Income levels** indicate: insurance coverage, care utilization patterns, hospital funding

**Limitations:**
- ZIP → MSA mapping is approximate (centroid distances)
- Population decay ignores actual travel patterns (uses Euclidean distance)
- Income assumes nearest MSA represents ZIP's socio-economic context
- Racial demographics excluded to avoid legal/ethical complications

### Population Stratification

```
General Population
│
├─ Subdomains
│   ├─ Medicare (elderly / disabled)
│   ├─ Psychiatric
│   └─ Other (general adult, non-Medicare)
│
└─ Demographics (from MSA / ZIP)
    ├─ State & Region
    ├─ Population density (urban/rural)
    └─ Average income
```

### Model Features
- State and census region
- Hospital mission type (women's, children's, cancer, rehabilitation)
- Ownership type (government, not-for-profit, proprietary)
- Estimated population density
- Estimated income per capita
- Domain-specific outcomes (general, psychiatric, Medicare)

## Business Objectives - Feasibility Assessment

### ✅ Achievable

**1. Improve Quality of Care**
- Data: HVBP measures, safety/readmission data, Medicare clinical outcomes
- Approach: Predictive models track year-over-year performance, identify at-risk hospitals

**2. Increase Patient Engagement**
- Data: HCAHPS and Person & Community Engagement measures
- Approach: Direct measurement of communication, responsiveness, satisfaction

**4. Increase Revenue Capture** (Partial)
- Data: HVBP, HAC, readmission reduction programs
- Approach: Identify hospitals at risk of losing reimbursement points

### ❌ Not Feasible with Current Data

**3. Reduce Cost of Care**
- Missing: Length of stay, staffing levels, procedure-specific costs
- Available: MSPB (limited spending info)
- Status: Cost reduction analysis would be speculative

**5. Provide Accurate Clinical Outcomes** (Limited)
- True clinical outcomes not included
- Partial proxy: IPFQR continuity-of-care measures for psychiatric subdomain

## Modeling Approach

### Strategy
1. **Predictive Framework**: Δy (year-over-year change) using AR(1)/delta multivariate regression
2. **Feature Alignment**: All features aligned by `hospital_id` and `fiscal_year`
3. **Lag Structure**: Previous-year measures used to model future changes
4. **Regularization**: Lasso/Ridge for high-dimensional feature selection

### Two-Pronged Analysis

**1. Spot Treatment / Immediate Focus**
- Identify hospitals where key measures likely to deteriorate
- Recommend short-term interventions: audits, follow-ups, staffing adjustments
- Goal: Triage based on forecasted risk

**2. Process & Structural Focus / Reflective Study**
- Recognize systemic patterns (e.g., rising readmissions, low psychiatric continuity)
- Propose follow-up studies to identify root causes
- Analyze workflows, handoffs, discharge procedures
- Goal: Long-term process improvements, reduce systemic risk

## Implementation Notes

### Database Configuration
Use singleton pattern with thread safety:
```python
connect(DB_PATH, check_same_thread=False)
```

### Visualization Tips
- Color by MSA, hospital type, or baseline rate to reveal patterns
- Size by population or outcome severity
- Patterns often emerge immediately with proper visual encoding

## Project Structure

```
├── data/
│   ├── raw/              # Original CMS datasets
│   ├── source/           # Cleaned source data
│   ├── augmented/        # MSA and region enrichment
│   ├── historical/       # 2021-2024 archives
│   └── zip/              # Geographic reference data
├── etl/
│   ├── source/           # Source table SQL definitions
│   ├── augmented/        # Demographic enrichment SQL
│   ├── prediction/       # Model training tables
│   └── recommendation/   # Output recommendation tables
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── etl/              # Extract, transform, load
│   ├── train/            # Model training & search
│   ├── transform/        # Feature engineering
│   ├── visualization/    # Charts and reports
│   └── recommendation/   # Actionable output generation
├── scripts/              # Automation & deployment
├── tests/                # Unit and integration tests
├── hooks/                # Pre-commit formatting & validation
└── metrics/              # Model performance outputs
```

## Next Steps

- [ ] Expand feature set with multi-year lag structures
- [ ] Multi-target modeling for simultaneous outcome prediction
- [ ] External validation on held-out facilities
- [ ] Real-time API for operational decision support
- [ ] Integration with hospital EHR systems
- [ ] Dashboard for non-technical stakeholders

## Data Refresh

**New data**: Pull metadata from CMS API, clean filenames, download into appropriate year directory

**Historical data**: Extract from ZIP archives, move to temp directory, run `_dev_filter_.sh` for processing
⚠️ **Note**: Several scripts should be rewritten and tested before production use

---

*This project analyzes Medicare hospital performance to improve quality of care, patient safety, and healthcare outcomes through predictive analytics and demographic context.*

### TODO: Measure Polarity Review


### Api Needs Haversine distance for the query
As above the 111 scaling rule does not apply as expected, might need to dig into the math
zipcode maps also update it could be that the zips have changed in those regions

### Aggregate Scores

For the psychiatric (IPFQR) measures, the current prototype applies a coarse
polarity rule (treating all rate/percent columns as if lower is better).
This is directionally useful for the demo but not clinically precise.

Planned improvement:
- Define a per-measure polarity map (e.g., restraint/seclusion rates = lower is better,
  follow-up / intervention / screening percentages = higher is better).
- Apply the same review to other program tables to ensure all composite scores
  consistently reflect “higher is better” from a quality-of-care perspective.

Due to time constraints for this prototype, this refinement is noted but not yet implemented.

Should check other tables and all modeled features as well
