# This program was developed through ChatGPT
# This code is only used to create a random test file following the format
# of the joined table created with CreatingDataset.sql for testing purposes.
# If the MIMIC data cannot be obtained, this file generates data with the same
# headers and formatting, but completely random data that does not match any
# real individual.

import csv
import random
from datetime import datetime, timedelta

# Number of fake rows to generate
NUM_ROWS = 10000

# CSV file path
OUTPUT_FILE = "fake_test.csv"

# Define header
HEADER = [
    "SUBJECT_ID",
    "HADM_ID",
    "HOSPITAL_EXPIRE_FLAG",
    "ADMITTIME",
    "ETHNICITY",
    "GENDER",
    "DOB",
    "AGE_AT_ADMISSION",
    "ICD9_CODE_1",
    "ICD9_CODE_2",
    "ICD9_CODE_3"
]

# Example ethnicities and ICD9 codes
ETHNICITIES = ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "HISPANIC/LATINO", "OTHER"]
GENDERS = ["M", "F"]
ICD9_CODES = [
    "4019", "25000", "486", "2724", "41401", "4280", "5849", "5990",
    "51881", "78079", "49390", "42731", "53081", "53081", "71590",
    "53081", "7242", "496", "5859", "2449", "71536", "2720", "2768"
]

def random_datetime(start_year=2015, end_year=2025):
    """Generate a random datetime between Jan 1 start_year and now."""
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31, 23, 59, 59)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)

def random_dob(admit_time):
    """Generate DOB based on ADMITTIME (age 0-89, 300 for masked elderly)."""
    age = random.randint(0, 90)
    year = admit_time.year - age
    month = random.randint(1, 12)
    day = random.randint(1, 28)  # safe for all months
    dob = datetime(year, month, day)
    return dob, 300 if age > 89 else age if age > 0 else 0

with open(OUTPUT_FILE, "w", newline="") as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)

    # Write header
    writer.writerow(HEADER)

    for _ in range(NUM_ROWS):
        subject_id = random.randint(10000, 99999)
        hadm_id = random.randint(200000, 299999)
        hospital_expire_flag = random.choice([0, 1])
        admittime = random_datetime()
        dob, age_at_admission = random_dob(admittime)

        ethnicity = random.choice(ETHNICITIES)
        gender = random.choice(GENDERS)

        # Pick up to 3 ICD9 codes (can be NULL)
        icd_codes = random.sample(ICD9_CODES, k=random.randint(1, 3))
        while len(icd_codes) < 3:
            icd_codes.append(None)

        writer.writerow([
            subject_id,
            hadm_id,
            hospital_expire_flag,
            admittime.strftime("%Y-%m-%d %H:%M:%S"),
            ethnicity,
            gender,
            dob.strftime("%Y-%m-%d %H:%M:%S"),
            age_at_admission,
            icd_codes[0],
            icd_codes[1],
            icd_codes[2]
        ])

print(f"Fake CSV generated: {OUTPUT_FILE}")
