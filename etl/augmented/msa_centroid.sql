create table msa_centroid (
    year INT,
    cbsafp INT,
    -- name -> msa_title
    msa_title TEXT,
    state_abbreviation TEXT,
    -- intptlat -> latitude
    -- intptlon  -> longitude
    latitude REAL,
    longitude REAL
);

