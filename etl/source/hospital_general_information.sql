CREATE TABLE hospital_general_information (
    submission_year INT,
    facility_id INT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,
    telephone_number TEXT,
    hospital_type TEXT,
    hospital_ownership TEXT,
    emergency_services TEXT,
    birthing_friendly_designation TEXT,
    hospital_overall_rating TEXT,
    hospital_overall_rating_footnote TEXT,
    -- measures differ b/c might not have enough data to be statistically valid
     
    -- mortality
    mort_group_measure_count INT,
    count_of_facility_mort_measures INT,
    count_of_mort_measures_better INT,
    count_of_mort_measures_no_different INT,
    count_of_mort_measures_worse INT,
    mort_group_footnote TEXT,
    
    -- safety
    safety_group_measure_count INT,
    count_of_facility_safety_measures INT,
    count_of_safety_measures_better INT,
    count_of_safety_measures_no_different INT,
    count_of_safety_measures_worse INT,
    safety_group_footnote TEXT,
    
    -- readmission
    readm_group_measure_count INT,
    count_of_facility_readm_measures INT,
    count_of_readm_measures_better INT,
    count_of_readm_measures_no_different INT,
    count_of_readm_measures_worse INT,
    readm_group_footnote TEXT,
    
    -- person and community engagement (patient experience)
    pt_exp_group_measure_count INT,
    count_of_facility_pt_exp_measures INT,
    pt_exp_group_footnote TEXT,
    
    -- total expense, medicare spending / beneficiary 
    te_group_measure_count INT,
    count_of_facility_te_measures INT,
    te_group_footnote TEXT
);

