CREATE TABLE ipfqr_quality_measures_facility (
    submission_year INT,
    -- IPFQR := Inpatient Psychiatric Facility Quality Reporting Program

    facility_id TEXT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,
    -- per 1000 means per 1000 patient hours

    -- mental health: hours of physical restraint use
    hbips_2_measure_description TEXT,
    hbips_2_overall_rate_per_1000 REAL,
    hbips_2_overall_num INT,
    hbips_2_overall_den INT,
    hbips_2_overall_footnote TEXT,

    -- mental health: hours of seclusion use
    hbips_3_measure_description TEXT,
    hbips_3_overall_rate_per_1000 REAL,
    hbips_3_overall_num INT,
    hbips_3_overall_den INT,
    hbips_3_overall_footnote TEXT,

    -- screening for metabolic disorders
    smd_measure_description TEXT,
    smd_percent REAL,
    smd_denominator INT,
    smd_footnote TEXT,

    -- alcohol use brief intervention provided or offered
    sub_2_2a_measure_description TEXT,
    sub_2_percent REAL,
    sub_2_denominator INT,
    sub_2_footnote TEXT,
    sub2_a_percent REAL,
    sub2_a_denominator INT,
    sub2_a_footnote TEXT,

    -- alcohol and other drug use disorder treatment provided or offered at discharge
    sub_3_3a_measure_description TEXT,
    sub_3_percent REAL,
    sub_3_denominator INT,
    sub_3_footnote TEXT,
    sub3_a_percent REAL,
    sub3_a_denominator INT,
    sub3_a_footnote TEXT,

    -- tobacco use treatment provided or offered at discharge
    tob_3_3a_measure_description TEXT,
    tob_3_percent REAL,
    tob_3_denominator INT,
    tob_3_footnote TEXT,
    tob_3a_percent REAL,
    tob_3a_denominator INT,
    tob_3a_footnote TEXT,

    -- transition record with specified elements received by discharged patients
    tr_1_measure_description TEXT,
    tr_1_percent REAL,
    tr_1_denominator INT,
    tr_1_footnote TEXT,

    start_date DATE,
    end_date DATE,

    -- follow-up after psychiatric hospitalization
    faph_measure_description TEXT,
    faph_30_percent REAL,
    faph_30_denominator INT,
    faph_30_footnote TEXT,
    faph_7_percent REAL,
    faph_7_denominator INT,
    faph_7_footnote TEXT,
    faph_measure_start_date DATE,
    faph_measure_end_date DATE,

    -- patients discharged from an ipf with all medications reconciled and communicated to the next care provider (medical continuity)
    medcont_measure_description TEXT,
    medcont_percent REAL,
    medcont_denominator INT,
    medcont_footnote TEXT,
    medcont_measure_start_date DATE,
    medcont_measure_end_date DATE,

    -- readmission rates
    readm_30_ipf_measure_description TEXT,
    readm_30_ipf_category TEXT,
    readm_30_ipf_denominator INT,
    readm_30_ipf_rate REAL,
    readm_30_ipf_lower_estimate REAL,
    readm_30_ipf_higher_estimate REAL,
    readm_30_ipf_footnote TEXT,
    readm_30_ipf_start_date DATE,
    readm_30_ipf_end_date DATE,

    -- influenza immunization rates
    imm_2_measure_description TEXT,
    imm_2_percent REAL,
    imm_2_denominator INT,
    imm_2_footnote TEXT,

    flu_season_start_date DATE,
    flu_season_end_date DATE
);
