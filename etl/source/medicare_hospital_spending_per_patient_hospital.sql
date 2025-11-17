-- NOTE: Was a replacement dataset
-- The Medicare Spending Per Beneficiary (MSPB) Measure shows whether Medicare spends more, less, or about the same for an episode of care (episode) at a specific hospital compared to all hospitals nationally.
-- An MSPB episode includes Medicare Part A and Part B payments for services provided by hospitals and other healthcare providers the 3 days prior to, during, and 30 days following a patient's inpatient stay. This measure evaluates hospitals' costs compared to the costs of the national median (or midpoint) hospital.
-- This measure takes into account important factors like patient age and health status (risk adjustment) and geographic payment differences (payment-standardization).

CREATE TABLE medicare_hospital_spending_per_patient_hospital (
    -- only contains Medicare hospital spending per patient (Medicare Spending per Beneficiary)
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

    score REAL,
    footnote TEXT,
    start_date DATE,
    end_date DATE
);
