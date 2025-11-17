CREATE TABLE complications_and_deaths_hospital (
    submission_year INT,
    facility_id INT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,
    telephone_number TEXT,

    -- ssi, mort, complications
    measure_id TEXT,
    measure_name TEXT,

    compared_to_national TEXT,
    denominator INT,
    score REAL,

    lower_estimate REAL,
    higher_estimate REAL,
    footnote TEXT,

    start_date DATE,
    end_date DATE
);
