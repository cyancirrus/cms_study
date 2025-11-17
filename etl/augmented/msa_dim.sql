-- table used for reconciliation to ensure the mappings align
create table msa_dim (
    year INT,
    -- CBSA Code -> cbsafp
    cbsafp INT,
    cbsa_title TEXT,
    -- STATE -> state_abbreviation
    state_abbreviation TEXT    
);

