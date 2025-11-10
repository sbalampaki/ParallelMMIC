/* Programmed by Caleb Griffy
   MySQL code for cleaning up MIMIC III datasets for use with our parallel research project.
   The datasets necessary must be obtained from MIMIC III itself, and will not be shared by us.
   However, a sample dataset that is formatted similarly can be obtained, but the values have been randomly generated and are meaningless. */

# Allow MySQL to access files
SET GLOBAL local_infile = 1;
SET PERSIST local_infile = 1;

# Create database for use
CREATE DATABASE IF NOT EXISTS mimic_data;
USE mimic_data;

# Create tables:
# Admissions
CREATE TABLE `admissions` (
  `ROW_ID` int NOT NULL,
  `SUBJECT_ID` int DEFAULT NULL,
  `HADM_ID` int DEFAULT NULL,
  `ADMITTIME` datetime DEFAULT NULL,
  `DISCHTIME` datetime DEFAULT NULL,
  `DEATHTIME` datetime DEFAULT NULL,
  `ADMISSION_TYPE` varchar(50) DEFAULT NULL,
  `ADMISSION_LOCATION` varchar(50) DEFAULT NULL,
  `DISCHARGE_LOCATION` varchar(50) DEFAULT NULL,
  `INSURANCE` varchar(255) DEFAULT NULL,
  `LANGUAGE` varchar(10) DEFAULT NULL,
  `RELIGION` varchar(50) DEFAULT NULL,
  `MARITAL_STATUS` varchar(50) DEFAULT NULL,
  `ETHNICITY` varchar(50) DEFAULT NULL,
  `EDREGTIME` datetime DEFAULT NULL,
  `EDOUTTIME` datetime DEFAULT NULL,
  `DIAGNOSIS` varchar(300) DEFAULT NULL,
  `HOSPITAL_EXPIRE_FLAG` tinyint DEFAULT NULL,
  `HAS_CHARTEVENTS_DATA` tinyint DEFAULT NULL,
  PRIMARY KEY (`ROW_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

# Diagnoses_ICD
CREATE TABLE `diagnoses_icd` (
  `ROW_ID` int NOT NULL,
  `SUBJECT_ID` int NOT NULL,
  `HADM_ID` int NOT NULL,
  `SEQ_NUM` int DEFAULT NULL,
  `ICD9_CODE` varchar(10) DEFAULT NULL,
  PRIMARY KEY (`ROW_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

# Patients
CREATE TABLE `patients` (
  `ROW_ID` int NOT NULL,
  `SUBJECT_ID` int DEFAULT NULL,
  `GENDER` varchar(5) DEFAULT NULL,
  `DOB` datetime DEFAULT NULL,
  `DOD` datetime DEFAULT NULL,
  `DOD_HOSP` datetime DEFAULT NULL,
  `DOD_SSN` datetime DEFAULT NULL,
  `EXPIRE_FLAG` varchar(5) DEFAULT NULL,
  PRIMARY KEY (`ROW_ID`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

# Read data into tables. You'll have to change the directory to match your own location for the files
# Admissions
LOAD DATA LOCAL INFILE 'C:/Users/caleb/Desktop/MIMICIII SQL Project/mimiciii/ADMISSIONS.csv/ADMISSIONS.csv'
INTO TABLE ADMISSIONS
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

# Diagnoses_ICD
LOAD DATA LOCAL INFILE 'C:/Users/caleb/Desktop/MIMICIII SQL Project/mimiciii/DIAGNOSES_ICD.csv/DIAGNOSES_ICD.csv'
INTO TABLE DIAGNOSES_ICD
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

# Patients
LOAD DATA LOCAL INFILE 'C:/Users/caleb/Desktop/MIMICIII SQL Project/mimiciii/PATIENTS.csv/PATIENTS.csv'
INTO TABLE PATIENTS
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 LINES;

# Create joined table
CREATE TABLE joined_admissions AS
SELECT
    a.SUBJECT_ID,
    a.HADM_ID,
    a.HOSPITAL_EXPIRE_FLAG,
    a.ADMITTIME,
    a.ETHNICITY,
    p.GENDER,
    p.DOB,
    TIMESTAMPDIFF(YEAR, p.DOB, a.ADMITTIME) AS AGE_AT_ADMISSION,
    d1.ICD9_CODE AS ICD9_CODE_1,
    d2.ICD9_CODE AS ICD9_CODE_2,
    d3.ICD9_CODE AS ICD9_CODE_3
FROM ADMISSIONS AS a
JOIN PATIENTS AS p
    ON a.SUBJECT_ID = p.SUBJECT_ID
LEFT JOIN DIAGNOSES_ICD AS d1
    ON a.HADM_ID = d1.HADM_ID AND d1.SEQ_NUM = 1
LEFT JOIN DIAGNOSES_ICD AS d2
    ON a.HADM_ID = d2.HADM_ID AND d2.SEQ_NUM = 2
LEFT JOIN DIAGNOSES_ICD AS d3
    ON a.HADM_ID = d3.HADM_ID AND d3.SEQ_NUM = 3;
    
# Uncomment this if you want to see the joined table:
# SELECT * FROM JOINED_ADMISSIONS;

# For saving, run this line before the rest of the code below:
SHOW VARIABLES LIKE 'secure_file_priv';
/* This displays the folder you have permission to create files in.
   Use that directory for the INTO OUTFILE line below, with the chosen csv name.
   When ready, uncomment the code below and run it.*/

# Save table as csv
/*
SELECT *
FROM joined_admissions
INTO OUTFILE 'C:/ProgramData/MySQL/MySQL Server 9.1/Uploads/project_data.csv'
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n';
*/