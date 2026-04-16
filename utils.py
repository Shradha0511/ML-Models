import pandas as pd
from sklearn.model_selection import train_test_split

def load_data():
    df = pd.read_csv("HR.csv")
    df = df.drop(columns=["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"])
    df["Attrition"] = (df["Attrition"] == "Yes").astype(int)
    
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)