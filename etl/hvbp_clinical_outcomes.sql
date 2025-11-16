CREATE TABLE hvbp_clinical_outcomes (
    submission_year INT,
    fiscal_year INT,
    facility_id INT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,

    -- All-Cause, Risk-Standardized Mortality Rate following procedure 30 day mortality rate
    -- mortality: Acute Myocardial Infarction (AMI) 30-day mortality rate
    mort_30_ami_achievement_threshold REAL,
    mort_30_ami_benchmark REAL,
    mort_30_ami_baseline_rate REAL,
    mort_30_ami_performance_rate REAL,
    mort_30_ami_achievement_points TEXT,
    mort_30_ami_improvement_points TEXT,
    mort_30_ami_measure_score TEXT,

    -- Heart Failure (HF) 
    mort_30_hf_achievement_threshold REAL,
    mort_30_hf_benchmark REAL,
    mort_30_hf_baseline_rate REAL,
    mort_30_hf_performance_rate REAL,
    mort_30_hf_achievement_points TEXT,
    mort_30_hf_improvement_points TEXT,
    mort_30_hf_measure_score TEXT,

    -- Pneumonia (PN) 
    mort_30_pn_achievement_threshold REAL,
    mort_30_pn_benchmark REAL,
    mort_30_pn_baseline_rate REAL,
    mort_30_pn_performance_rate REAL,
    mort_30_pn_achievement_points TEXT,
    mort_30_pn_improvement_points TEXT,
    mort_30_pn_measure_score TEXT,

    -- mortality: Chronic Obstructive Pulmonary Disease (COPD)
    mort_30_copd_achievement_threshold REAL,
    mort_30_copd_benchmark REAL,
    mort_30_copd_baseline_rate REAL,
    mort_30_copd_performance_rate REAL,
    mort_30_copd_achievement_points TEXT,
    mort_30_copd_improvement_points TEXT,
    mort_30_copd_measure_score TEXT,

    -- mortality: Coronary Artery Bypass Graft Surgery (CAPG)
    mort_30_cabg_achievement_threshold REAL,
    mort_30_cabg_benchmark REAL,
    mort_30_cabg_baseline_rate REAL,
    mort_30_cabg_performance_rate REAL,
    mort_30_cabg_achievement_points TEXT,
    mort_30_cabg_improvement_points TEXT,
    mort_30_cabg_measure_score TEXT,

    -- complication rate : usually 90 days post discharge but unknown
    -- a complication rate is a risk-adjusted proportion of patients who experience certain serious but preventable adverse events after a specific procedure or hospitalization.
    comp_hip_knee_achievement_threshold REAL,
    comp_hip_knee_benchmark REAL,
    comp_hip_knee_baseline_rate REAL,
    comp_hip_knee_performance_rate REAL,
    comp_hip_knee_achievement_points TEXT,
    comp_hip_knee_improvement_points TEXT,
    comp_hip_knee_measure_score TEXT
);

