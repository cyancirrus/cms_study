CREATE TABLE ipfqr_quality_measures_facility (
    submission_year INT,
    -- IPFQR := Inpatient Psychiatric Facility Quality Reporting Program

    facility_id INT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,
    -- per 1000 means per 1000 patient hours

    -- mental health: hours of physical restraint use
    hbips2_measure_description TEXT,
    hbips2_overall_rate_per_1000 REAL,
    hbips2_overall_num INT,
    hbips2_overall_den INT,
    hbips2_overall_footnote TEXT,

    -- mental health: hours of seclusion use
    hbips3_measure_description TEXT,
    hbips3_overall_rate_per_1000 REAL,
    hbips3_overall_num INT,
    hbips3_overall_den INT,
    hbips3_overall_footnote TEXT,

    -- screening for metabolic disorders
    smd_measure_description TEXT,
    smd_percent REAL,
    smd_denominator INT,
    smd_footnote TEXT,

    -- alcohol use brief intervention provided or offered
    sub2_2a_measure_description TEXT,
    sub2_percent REAL,
    sub2_denominator INT,
    sub2_footnote TEXT,
    sub2a_percent REAL,
    sub2a_denominator INT,
    sub2a_footnote TEXT,

    -- alcohol and other drug use disorder treatment provided or offered at discharge
    sub3_3a_measure_description TEXT,
    sub3_percent REAL,
    sub3_denominator INT,
    sub3_footnote TEXT,
    sub3a_percent REAL,
    sub3a_denominator INT,
    sub3a_footnote TEXT,

    -- tobacco use treatment provided or offered at discharge
    tob3_3a_measure_description TEXT,
    tob3_percent REAL,
    tob3_denominator INT,
    tob3_footnote TEXT,
    tob3a_percent REAL,
    tob3a_denominator INT,
    tob3a_footnote TEXT,

    -- transition record with specified elements received by discharged patients
    tr1_measure_description TEXT,
    tr1_percent REAL,
    tr1_denominator INT,
    tr1_footnote TEXT,

    start_date DATE,
    end_date DATE,

    -- follow-up after psychiatric hospitalization
    faph_measure_description TEXT,
    faph30_percent REAL,
    faph30_denominator INT,
    faph30_footnote TEXT,
    faph7_percent REAL,
    faph7_denominator INT,
    faph7_footnote TEXT,
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
    readm30_ipf_measure_description TEXT,
    readm30_ipf_category TEXT,
    readm30_ipf_denominator INT,
    readm30_ipf_rate REAL,
    readm30_ipf_lower_estimate REAL,
    readm30_ipf_higher_estimate REAL,
    readm30_ipf_footnote TEXT,
    readm30_ipf_start_date DATE,
    readm30_ipf_end_date DATE,

    -- influenza immunization rates
    imm2_measure_description TEXT,
    imm2_percent REAL,
    imm2_denominator INT,
    imm2_footnote TEXT,

    flu_season_start_date DATE,
    flu_season_end_date DATE
);
