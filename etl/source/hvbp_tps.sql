CREATE TABLE hvbp_tps (
    submission_year INT,
    fiscal_year INT,
    facility_id TEXT,
    facility_name TEXT,
    address TEXT,
    city_town TEXT,
    state CHAR(2),
    zip_code CHAR(5),
    county_parish TEXT,

    -- Clinical Outcomes	Mortality, complications, readmission rates	30-day mortality for heart attack, heart failure, pneumonia	~25%
    unweighted_normalized_clinical_outcomes_domain_score REAL,
    weighted_normalized_clinical_outcomes_domain_score REAL,

    -- Person and Community Engagement	Patient experience (HCAHPS survey)	“Would recommend hospital,” nurse communication	~25%
    unweighted_person_and_community_engagement_domain_score REAL,
    weighted_person_and_community_engagement_domain_score REAL,

    -- Safety	Healthcare-associated infections	CLABSI, CAUTI, MRSA, C. diff	~25%
    unweighted_normalized_safety_domain_score REAL,
    weighted_safety_domain_score REAL,

    -- Efficiency and Cost Reduction	Medicare spending per beneficiary	Risk-adjusted cost per episode	~25%
    unweighted_normalized_efficiency_and_cost_reduction_domain_score REAL,
    weighted_efficiency_and_cost_reduction_domain_score REAL,

    total_performance_score REAL
);
