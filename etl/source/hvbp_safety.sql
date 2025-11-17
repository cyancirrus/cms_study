CREATE TABLE hvbp_safety (
    submission_year INT,
    fiscal_year INT,
    facility_id TEXT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,
    -- HAI-1: Measures Central Line-Associated Bloodstream Infections (CLABSI).

    hai1_achievement_threshold REAL,
    hai1_benchmark REAL,
    hai1_baseline_rate REAL,
    hai1_performance_rate REAL,
    hai1_achievement_points TEXT,
    hai1_improvement_points TEXT,
    hai1_measure_score TEXT,

    -- HAI-2: Measures Catheter-Associated Urinary Tract Infections (CAUTI).
    hai2_achievement_threshold REAL,
    hai2_benchmark REAL,
    hai2_baseline_rate REAL,
    hai2_performance_rate REAL,
    hai2_achievement_points TEXT,
    hai2_improvement_points TEXT,
    hai2_measure_score TEXT,

    combined_ssi_measure_score TEXT,

    -- HAI-3: Measures MRSA (Methicillin-resistant Staphylococcus aureus) infections.
    hai3_achievement_threshold REAL,
    hai3_benchmark REAL,
    hai3_baseline_rate REAL,
    hai3_performance_rate REAL,
    hai3_achievement_points TEXT,
    hai3_improvement_points TEXT,
    hai3_measure_score TEXT,

    -- HAI-4: Measures CDI (Clostridioides difficile infections).
    hai4_achievement_threshold REAL,
    hai4_benchmark REAL,
    hai4_baseline_rate REAL,
    hai4_performance_rate REAL,
    hai4_achievement_points TEXT,
    hai4_improvement_points TEXT,
    hai4_measure_score TEXT,

    -- HAI-5: Methicillin-resistant Staphylococcus aureus (MRSA) Blood Laboratory-identified Events (bloodstream infections).
    hai5_achievement_threshold REAL,
    hai5_benchmark REAL,
    hai5_baseline_rate REAL,
    hai5_performance_rate REAL,
    hai5_achievement_points TEXT,
    hai5_improvement_points TEXT,
    hai5_measure_score TEXT,

    -- HAI-6: Clostridium difficile (C. diff) Laboratory-identified Events (intestinal infections).
    hai6_achievement_threshold REAL,
    hai6_benchmark REAL,
    hai6_baseline_rate REAL,
    hai6_performance_rate REAL,
    hai6_achievement_points TEXT,
    hai6_improvement_points TEXT,
    hai6_measure_score TEXT
);
