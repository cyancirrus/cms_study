create table msa_statistics (
    -- 2023 was the latest year this will be static data refresh schedule unknown
    year INT,
    -- msa_id -> cbsafp
    cbsafp INT,
    linecode,
    description,
    -- 2023 -> metric
    metric
);
