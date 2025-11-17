CREATE TABLE hcahps_hospital (
    submission_year INT,
    -- HCAHPS := Hospital Consumer Assessment of Healthcare Providers and Systems
    facility_id TEXT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,
    telephone_number TEXT,

    -- HCAHPS is the national, standardized survey given to recently discharged patients about their hospital experience, while the patient survey
    -- star rating is a public-facing 5-star rating system that summarizes the results from that HCAHPS survey.
    -- Essentially, HCAHPS is the source data, and the patient survey star rating is a tool to make that data easier for consumers to understand and compare hospitals.

    hcahps_measure_id TEXT,
    hcahps_question TEXT,
    hcahps_answer_description TEXT,

    patient_survey_star_rating TEXT,
    patient_survey_star_rating_footnote TEXT,

    hcahps_answer_percent REAL,
    hcahps_answer_percent_footnote TEXT,
    hcahps_linear_mean_value REAL,

    number_of_completed_surveys INT,
    number_of_completed_surveys_footnote TEXT,

    survey_response_rate_percent REAL,
    survey_response_rate_percent_footnote TEXT,

    start_date DATE,
    end_date DATE
);
