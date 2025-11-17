CREATE TABLE unplanned_hospital_visits_hospital (
    submission_year INT,
    facility_id TEXT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,
    telephone_number TEXT,

    -- should be measures like readmission infection mortality etc
    measure_id TEXT,
    measure_name TEXT,

    compared_to_national TEXT,
    denominator REAL,
    score REAL,

    lower_estimate REAL,
    higher_estimate REAL,

    number_of_patients INT,
    number_of_patients_returned INT,
    footnote TEXT,

    start_date DATE,
    end_date DATE
);
