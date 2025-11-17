-- 2023 was the latest year this will be static data refresh schedule unknown
create table msa_statistics (
    submission_year INT,
    -- msa_id -> cbsafp
    cbsafp INT,
    linecode TEXT,
    description TEXT,
    -- 2023 -> metric
    metric REAL
);
