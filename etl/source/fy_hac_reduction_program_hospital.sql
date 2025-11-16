CREATE TABLE fy_hac_reduction_program_hospital (
    submission_year INT,
    -- HAC:= Hospital-Acquired Condition Reduction Program
    -- HAC Reduction is Medicare specific
    -- Medicare = federal health insurance for the elderly (65+) and certain disabled populations.
    -- Medicaid = state + federal program for low-income individuals.
    -- w-zscore := Weighted Z Score ~ (observed - expected)/standard_error;

    facility_name TEXT,
    facility_id INT,
    state CHAR(2),
    fiscal_year INT,
    
    -- PSI90 is a composite of 10 different patient safety indicators, including events like:
    --     Pressure ulcers (PSI 03)
    --     Postoperative sepsis (PSI 13)
    --     In-hospital falls with hip fracture (PSI 08)
    --     Postoperative respiratory failure (PSI 11)
    --     Perioperative pulmonary embolism or deep vein thrombosis (PSI 12
    psi90_composite_value REAL,
    psi90_composite_value_footnote TEXT,
    psi90_w_z_score REAL,
    psi90_w_z_footnote TEXT,
    psi90_start_date DATE,
    psi90_end_date DATE,
    -- SIR :: Standardized Infection Ratio
    
    -- Central Line-Associated Bloodstream Infections (CLABSI
    clabsi_sir REAL,
    clabsi_sir_footnote TEXT,
    clabsi_w_z_score REAL,
    clabsi_w_z_footnote TEXT,
    
    -- Catheter-Associated Urinary Tract Infection (CAUTI)
    cauti_sir REAL,
    cauti_sir_footnote TEXT,
    cauti_w_z_score REAL,
    cauti_w_z_footnote TEXT,
    
    -- Surgical sight infections
    ssi_sir REAL,
    ssi_sir_footnote TEXT,
    ssi_w_z_score REAL,
    ssi_w_z_footnote TEXT,
    
    -- Infection from Clostridioides difficile infections
    cdi_sir REAL,
    cdi_sir_footnote TEXT,
    cdi_w_z_score REAL,
    cdi_w_z_footnote TEXT,
    
    -- Methicillin-Resistant Staphylococcus Aureus Standardized Infection Ratio
    mrsa_sir REAL,
    mrsa_sir_footnote TEXT,
    mrsa_w_z_score REAL,
    mrsa_w_z_footnote TEXT,

    -- hospital associated infections 
    hai_measures_start_date DATE,
    hai_measures_end_date DATE,
   

    -- Hospital-Acquired Condition Reduction Program
    -- Centers for Medicare & Medicaid Services | CMS (.gov)
    -- CMS uses the Total HAC Score to determine the worst-performing quartile of all subsection (d) hospitals based on data for six quality measures:.
    total_hac_score REAL,
    total_hac_score_footnote TEXT,

    -- if in bottom 25% quartile then recieves 0.99 * medicare payments (1% less payment, average margin is something like 1-5% margin)
    payment_reduction REAL,
    payment_reduction_footnote TEXT
);

