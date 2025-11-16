CREATE TABLE fy_hospital_readmissions_reduction_program_hospital (
    -- readmissions
    submission_year INT,
    facility_name TEXT,
    facility_id INT,
    state CHAR(2),

    measure_name TEXT,
    number_of_discharges INT,
    footnote TEXT,

    excess_readmission_ratio REAL,
    
    predicted_readmission_rate REAL,
    expected_readmission_rate REAL,

    number_of_readmissions INT,

    start_date DATE,
    end_date DATE
);

