import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fake_test.csv")

died_df = df[df["HOSPITAL_EXPIRE_FLAG"] == 1]

print("Patients who died in hospital:")
print(died_df.head())

grouped = (
    died_df
    .groupby("ICD9_CODE_1")
    .size()
    .reset_index(name="num_patients")
    .sort_values("num_patients", ascending=False)
)

print("\nNumber of deceased patients per ICD9_CODE_1:")
print(grouped)


plt.figure(figsize=(12,6))
plt.bar(grouped["ICD9_CODE_1"].astype(str), grouped["num_patients"])
plt.title("Number of Deceased Patients by ICD9_CODE_1")
plt.xlabel("ICD9_CODE_1")
plt.ylabel("Number of Patients")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,8))
plt.pie(
    grouped["num_patients"], 
    labels=grouped["ICD9_CODE_1"].astype(str),
    autopct="%1.1f%%",
    startangle=140
)
plt.title("Distribution of Deceased Patients by ICD9_CODE_1")
plt.tight_layout()
plt.show()

group_icd9_1 = (
    df.groupby("ICD9_CODE_1")
      .size()
      .reset_index(name="num_patients")
      .sort_values("num_patients", ascending=False)
)

print("\nPatients per ICD9_CODE_1:")
print(group_icd9_1)

death_stats = (
    df.groupby("ICD9_CODE_1")
      .agg(
          total_patients=("HOSPITAL_EXPIRE_FLAG", "size"),
          total_deaths=("HOSPITAL_EXPIRE_FLAG", "sum")
      )
)

death_stats["death_rate"] = death_stats["total_deaths"] / death_stats["total_patients"]

death_stats = death_stats.sort_values("death_rate", ascending=False)

print("\nDeath rate per ICD9_CODE_1:")
print(death_stats)

plt.figure(figsize=(12,6))
plt.bar(death_stats.index.astype(str), death_stats["death_rate"])
plt.title("Death Rate per ICD9_CODE_1")
plt.xlabel("ICD9_CODE_1")
plt.ylabel("Death Rate (0 to 1)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
plt.bar(death_stats.index.astype(str), death_stats["total_patients"])
plt.title("Number of Patients per ICD9_CODE_1")
plt.xlabel("ICD9_CODE_1")
plt.ylabel("Patient Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


plt.figure(figsize=(12,6))
plt.bar(death_stats.index.astype(str), death_stats["total_deaths"])
plt.title("Number of Deaths per ICD9_CODE_1")
plt.xlabel("ICD9_CODE_1")
plt.ylabel("Deaths (HOSPITAL_EXPIRE_FLAG = 1)")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

