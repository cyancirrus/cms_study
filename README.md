# DATA NOTES
- heavily consider grabbing 2024 data Provider value of care doesn't exist Downloaded https://data.cms.gov/provider-data/dataset/rrqw-56er :: Medicare Spending Per Beneficiary

- Data definition here :: https://data.cms.gov/provider-data/sites/default/files/data_dictionaries/hospital/HOSPITAL_Data_Dictionary.pdf
- Create exclusion column for "too little data" if it's a rate 
- do a quick check that like end_date = fiscal year, it's not perfect but i don't have a lot of time and there isn't monthly data to like pull back effects



# IMPORTANT DEFNS
-- CMS := Center for Medicare and Medicaid Services
-- HVBP := Hospital Value-Based Purchasing.
-- HAC:= Hospital-Acquired Condition Reduction Program
-- HAC Reduction is Medicare specific
-- Medicare = federal health insurance for the elderly (65+) and certain disabled populations.
-- Medicaid = state + federal program for low-income individuals.
-- HCAHPS := Hospital Consumer Assessment of Healthcare Providers and Systems
-- IPFQR := Inpatient Psychiatric Facility Quality Reporting Program

# HOW TO GET DATA

* new data * 
- hit the metadata api :: curl  "https://data.cms.gov/provider-data/api/1/metastore/schemas/dataset/items/avtz-f2ge"
- clean it and make sure like no forward slashes
- download data into the year


* old data * 
- need to extract from zip here "https://data.cms.gov/provider-data/archived-data/hospitals" for last year
- download onto machine mv to temp directory unzip the files using _dev_filter_.sh ./data/historical/year; // this really should be rewritten and tested prior 



## fiscal_year
- October 1 is inclusive, September 30 is inclusive. So a measure dated Sep 30, 2025 still belongs to FY25.
- Any timestamp after Sep 30, 2025 23:59:59 would roll over into FY26.

FY2025 = Oct 1, 2024 → Sep 30, 2025.


# DATA AUGMENTATION

<!-- // Data here seems much better -->
<!-- https://github.com/Ro-Data/Ro-Census-Summaries-By-Zipcode -->


## MSA Latitude Longitude
https://data-usdot.opendata.arcgis.com/datasets/usdot::core-based-statistical-areas/about


## MSA statistics
https://www.bea.gov/data/income-saving/personal-income-county-metro-and-other-areas

## MSA IDs // 
https://www.census.gov/geographies/reference-maps/2020/geo/cbsa.html
https://www2.census.gov/programs-surveys/metro-micro/geographies/reference-files/2015/delineation-files/list1.xlsx

## MSA Reasoning
- Population influence: decay as exp(-distance^2 / scale) from nearest MSA centroid
    - optional: sum top k nearest MSAs weighted by this decay
    - Population influence = Σ_i Population_i * exp(-distance_i^2 / (2 * σ^2))

- Income: take the nearest MSA’s median or per-capita income
    - used as a static socio-economic proxy


### Why

// allows us to see like demographics beyond state / city / county which would be too many like params and too few datas

* POPULATION DENSITY *
Urban vs rural, 
- more specialty doctors
- more people wish to live in said city
- higher competition for jobs etc..

* INCOME *
- Higher-income populations:
    - More likely to have employer-provided/private insurance
    - Better coverage, higher ability to pay for elective care
- Lower-income populations:
    - May delay care
    - Avoid specialty visits
    - Use ERs for primary care
- Hospital funding:
    - Local/state/federal taxes subsidize public hospitals
    - Higher local revenue → better infrastructure



### Notes / Limitations
- ZIP→MSA mapping approximate; using centroid distances
- Population decay ignores travel patterns; Euclidean distance used
- Income feature assumes nearest MSA is representative of ZIP’s socio-economic context
- Racial demographics excluded to avoid legal/ethical complications

## Mental Stratification
General Population
│
├─ Subdomains
│   ├─ Medicare (elderly / disabled)
│   ├─ Psychiatric
│   └─ Other (general adult, non-Medicare)
│
└─ Demographics (from MSA / ZIP)
    ├─ State
    ├─ Population density (urban/rural)
    └─ Average income

features :=
- state
- hospital mission type (womens, children, cancer, rehabilitation)
- ownership / payment type (government, not for profit, proprietary)
- estimated pop density
- estimated income per individual (taxes should average, even if not perfect, like is data i have)
- general pop x psychiatric x medicare


x_{t+1} = f(x_{t-1};

## Immediate vs like predictive actionability
1) Spot Treatment / Immediate Focus
Use your predictive model to identify hospitals or departments where key measures (readmissions, continuity-of-care, safety metrics) are likely to deteriorate.
Recommend short-term interventions or monitoring, e.g., targeted audits, patient follow-ups, or staffing adjustments.
Goal: mitigate the worst outcomes now — basically triage based on forecasted risk.

2) Process & Structural Focus / Reflective Study
Recognize that trends like rising readmissions or low psychiatric continuity-of-care aren’t isolated; they reflect latent systemic issues.
Propose follow-up studies to identify predictors or underlying processes driving these outcomes. For instance, analyze workflows, handoffs, or discharge procedures that correlate with readmission.
Goal: guide long-term process improvements that reduce systemic risk rather than treating individual symptoms.

# Enable virtual environment
source ../bin/activate

# NEXT STEPS
Data Considerations:
Prior-year data (2023, 2024) is needed for predictive modeling. These can be loaded from ZIP archives; otherwise, analysis would be limited to correlation-based insights.
Population density and median income, combined with state information, provide a reasonable proxy for demographic context.


Questions:
    Improve quality of care
    Increase patient engagement
    Reduce cost of care
    Increase revenue capture
    Provide accurate and timely clinical outcomes



1) Feasibility of Business Objectives:
Improve Quality of Care: Achievable using HVBP measures, safety/readmission data, and Medicare clinical outcomes. Predictive models can track YoY performance and identify hospitals with potential quality issues.

2) Increase Patient Engagement: Achievable via HCAHPS and Person & Community Engagement measures. While perception-based, these directly capture engagement, communication, responsiveness, and patient satisfaction.

3) Reduce Cost of Care: Not feasible with the current dataset. While MSPB provides some spending information, detailed operational metrics like length of stay, staffing, and procedure costs are missing, making cost reduction analysis speculative.

4) Increase Revenue Capture: Partially feasible. Data from HVBP, HAC, and readmission reduction programs can highlight hospitals at risk of losing points or reimbursements, helping management prioritize interventions to safeguard revenue.

5) Provide Accurate and Timely Clinical Outcomes: Limited feasibility. True clinical outcomes are not included, but psychiatric continuity-of-care measures (IPFQR) offer a partial proxy within that subdomain.

### Normalizations

## Complications and Deaths

// even more uncertain but think these are 100
COMP_HIP_KNEE|Rate of complications for hip/knee replacement patients

// ratio * 100
Hybrid_HWM|Hybrid Hospital-Wide All-Cause Risk Standardized Mortality Rate

// think these are per 100
MORT_30_AMI|Death rate for heart attack patients
MORT_30_CABG|Death rate for CABG surgery patients
MORT_30_COPD|Death rate for COPD patients
MORT_30_HF|Death rate for heart failure patients
MORT_30_PN|Death rate for pneumonia patients
MORT_30_STK|Death rate for stroke patients

// PSIS are per 1000 
PSI_03|Pressure ulcer rate
PSI_04|Death rate among surgical inpatients with serious treatable complications
PSI_06|Iatrogenic pneumothorax rate
PSI_08|In-hospital fall-associated fracture rate
PSI_09|Postoperative hemorrhage or hematoma rate
PSI_10|Postoperative acute kidney injury requiring dialysis rate
PSI_11|Postoperative respiratory failure rate
PSI_12|Perioperative pulmonary embolism or deep vein thrombosis rate
PSI_13|Postoperative sepsis rate
PSI_14|Postoperative wound dehiscence rate
PSI_15|Abdominopelvic accidental puncture or laceration rate
PSI_90|CMS Medicare PSI 90: Patient safety and adverse events composite



this one is extremely different scale, and has a mean 175 - it's per 1000 so need to divide it by 1000
"Death rate among surgical inpatients with serious treatable complications",

# Data Enrichment & Modeling Notes

**Current focus:** Predict Medicare-related hospital mortality/complication measures (Δy year-over-year).

## Potential Predictive Features

- **Hospital outcomes & complications**  
  - `Complications_and_Deaths-Hospital.csv`  
  - Includes AMI, HF, PN, COPD, CABG, hip/knee complications, PSI measures  
  - Use raw rates, aggregates, or weighted scores  

- **Safety & quality programs**  
  - HAC reduction (`FY_2025_HAC_Reduction_Program_Hospital.csv`)  
  - Readmission rates (`FY_2025_Hospital_Readmissions_Reduction_Program_Hospital.csv`)  

- **Patient experience**  
  - HCAHPS scores (`HCAHPS-Hospital.csv`, `PCH_HCAHPS_HOSPITAL.csv`)  

- **Hospital financials**  
  - Medicare spending per patient (`Medicare_Hospital_Spending_Per_Patient-Hospital.csv`)  

- **Care process measures**  
  - Timely and effective care (`Timely_and_Effective_Care-Hospital.csv`)  

- **Demographics / geography**  
  - MSA, population density, median income (via RUCA/ZIP/MSA mapping)  

## Approach

- Align all features by `hospital_id` & `fiscal_year`.  
- Lag previous-year measures to model Δy (year-over-year change).  
- Normalize/scale measures as needed.  
- Feed into AR(1)/delta multivariate regression for predictive modeling.  

**Goal:** Build a robust, interpretable model that uses prior outcomes and hospital context to forecast performance changes.

