CREATE TABLE hvbp_efficiency_and_cost_reduction (
    submission_year INT,
    -- mspb : medicare spending per beneficiary
    fiscal_year INT,
    facility_id TEXT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,

    -- medicare spending per beneficiary
    mspb1_achievement_threshold REAL,
    mspb1_benchmark REAL,
    mspb1_baseline_rate REAL,
    mspb1_performance_rate REAL,

    mspb1_achievement_points TEXT,
    mspb1_improvement_points TEXT,
    mspb1_measure_score TEXT
);
