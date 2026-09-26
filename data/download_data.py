import os
import urllib.request
import pandas as pd

DATA_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "diabetes_data.csv")

COLUMN_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

def download_and_save_data():
    print(f"Downloading dataset from: {DATA_URL}")
    df = pd.read_csv(DATA_URL, header=None, names=COLUMN_NAMES)
    
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Dataset successfully saved to: {OUTPUT_PATH}")
    print(f"\nShape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    assert df.shape == (768, 9), f"Expected shape (768, 9), got {df.shape}"
    print("\nVerification passed: Exactly 768 rows and 9 columns.")

if __name__ == "__main__":
    download_and_save_data()
