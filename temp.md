target = 10001
select * from zip_demographics where zip_code = 10001 limit 20;


"Facility ID",
"Facility Name",
"Address",
"City/Town",
"State",
"ZIP Code",
"County/Parish",
"HBIPS-2 Measure Description",
"HBIPS-2 Overall Rate Per 1000",
"HBIPS-2 Overall Num",
"HBIPS-2 Overall Den",
"HBIPS-2 Overall Footnote",
"HBIPS-3 Measure Description",
"HBIPS-3 Overall Rate Per 1000",
"HBIPS-3 Overall Num",
"HBIPS-3 Overall Den",
"HBIPS-3 Overall Footnote",
"SMD Measure Description",
"SMD %",
"SMD Denominator",
"SMD Footnote",
"SUB-2/-2a Measure Description",
"SUB-2 %",
"SUB-2 Denominator",
"SUB-2 Footnote",
"SUB-2a %",
"SUB-2a Denominator",
"SUB-2a Footnote",
"SUB-3/-3a Measure Description",
"SUB-3 %",
"SUB-3 Denominator",
"SUB-3 Footnote",
"SUB-3a %",
"SUB-3a Denominator",
"SUB-3a Footnote",
"TOB-3/-3a Measure Description",
"TOB-3 %",
"TOB-3 Denominator",
"TOB-3 Footnote",
"TOB-3a %",
"TOB-3a Denominator",
"TOB-3a Footnote",
"TR-1 Measure Description",
"TR-1 %",
"TR-1 Denominator",
"TR-1 Footnote",
"Start Date",
"End Date",
"FAPH Measure Description",
"FAPH-30 %",
"FAPH-30 Denominator",
"FAPH-30 Footnote",
"FAPH-7 %",
"FAPH-7 Denominator",
"FAPH-7 Footnote",
"FAPH Measure Start Date",
"FAPH Measure End Date",
"MedCont Measure Desc",
"MedCont %",
"MedCont Denominator",
"MedCont Footnote",
"MedCont Measure Start Date",
"MedCont Measure End Date",
"READM-30-IPF Measure Desc",
"READM-30-IPF Category",
"READM-30-IPF Denominator",
"READM-30-IPF Rate",
"READM-30-IPF Lower Estimate",
"READM-30-IPF Higher Estimate",
"READM-30-IPF Footnote",
"READM-30-IPF Start Date",
"READM-30-IPF End Date",
"IMM-2 Measure Description",
"IMM-2 %",
"IMM-2 Denominator",
"IMM-2 Footnote",
"Flu Season Start Date",
"Flu Season End Date"
;


SELECT
    submission_year,
    AVG(hbips_2_overall_rate_per_1000) AS hbips2_average,
    AVG(hbips_3_overall_rate_per_1000) AS hbips3_average,
    AVG(smd_percent) AS smd_average,
    AVG(sub_2_percent) AS sub2_average,
    AVG(sub_3_percent) AS sub3_average,
    AVG(tob_3_percent) AS tob3_average,
    AVG(tob_3a_percent) AS tob3a_average,
    AVG(tr_1_percent) AS tr1_average,
    AVG(imm_2_percent) AS imm2_average,
    AVG(readm_30_ipf_rate) AS readm30_average
FROM
    ipfqr_quality_measures_facility
GROUP BY
    submission_year
;


SELECT
    submission_year,
    sum(hbips_2_overall_rate_per_1000 IS NOT NULL) AS hbips2_count,
    sum(hbips_3_overall_rate_per_1000 IS NOT NULL) AS hbips3_count,
    sum(smd_percent IS NOT NULL) AS smd_count,
    sum(sub_2_percent IS NOT NULL) AS sub2_count,
    sum(sub_3_percent IS NOT NULL) AS sub3_count,
    sum(tob_3_percent IS NOT NULL) AS tob3_count,
    sum(tob_3a_percent IS NOT NULL) AS tob3a_count,
    sum(tr_1_percent IS NOT NULL) AS tr1_count,
    sum(imm_2_percent IS NOT NULL) AS imm2_count,
    sum(readm_30_ipf_rate IS NOT NULL) AS readm30_count
FROM
    ipfqr_quality_measures_facility
INNER JOIN
    zip_demographics
       ON  zip_demographics.zip_code = ipfqr_quality_measures_facility.zip_code
       ON  zip_demographics.submission_year = ipfqr_quality_measures_facility.zip_code
GROUP BY
    submission_year
;
<!-- INNER JOIN -->
<!--     facility_zip_code -->
<!--        ON  facility_zip_code.zip_code = ipfqr_quality_measures_facility.zip_code -->






// missing for 2022, 2023
    sum(faph_7_percent IS NOT NULL) AS faph7_count,
    sum(medcont_percent IS NOT NULL) AS medcont_count,

hbips_2_overall_rate_per_1000 IS NOT NULL
AND hbips_3_overall_rate_per_1000 IS NOT NULL
AND smd_percent IS NOT NULL
AND sub_2_percent IS NOT NULL
AND sub_3_percent IS NOT NULL
AND tob_3_percent IS NOT NULL
AND tob_3a_percent IS NOT NULL
AND tr_1_percent IS NOT NULL
AND imm_2_percent IS NOT NULL
AND readm_30_ipf_rate IS NOT NULL



SELECT
    submission_year,
    sum(hbips_2_overall_rate_per_1000 IS NOT NULL) AS hbips2_count,
    sum(hbips_3_overall_rate_per_1000 IS NOT NULL) AS hbips3_count,
    sum(smd_percent IS NOT NULL) AS smd_count,
    sum(sub_2_percent IS NOT NULL) AS sub2_count,
    sum(sub_3_percent IS NOT NULL) AS sub3_count,
    sum(tob_3_percent IS NOT NULL) AS tob3_count,
    sum(tob_3a_percent IS NOT NULL) AS tob3a_count,
    sum(tr_1_percent IS NOT NULL) AS tr1_count,
    sum(imm_2_percent IS NOT NULL) AS imm2_count,
    sum(readm_30_ipf_rate IS NOT NULL) AS readm30_count
FROM
    ipfqr_quality_measures_facility
GROUP BY
    submission_year
;

