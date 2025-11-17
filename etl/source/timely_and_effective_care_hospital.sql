CREATE TABLE timely_and_effective_care_hospital (
    submission_year INT,
    facility_id TEXT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,
    telephone_number TEXT,
    condition TEXT,
    -- should be measures like readmission infection mortality etc
    measure_id TEXT,
    measure_name TEXT,
    score TEXT,
    sample TEXT,
    footnote TEXT,
    start_date DATE,
    end_date DATE
);
