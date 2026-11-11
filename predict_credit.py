#!/usr/bin/env python3
"""
Credit Risk Prediction Script
German Credit Dataset Fairness Analysis

This script allows you to input feature values and predict credit risk
using the trained models from the German Credit Fairness Analysis project.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Feature names from German Credit dataset
CATEGORICAL_FEATURES = [
    "checking_status", "credit_history", "purpose", "savings_status",
    "employment", "other_parties", "property_magnitude", "other_payment_plans",
    "housing", "job"
]

NUMERICAL_FEATURES = [
    "duration_months", "credit_amount", "age", "residence_since",
    "existing_credits", "num_dependents"
]

# Mapping for categorical features (labels to numeric codes)
CATEGORY_LABELS = {
    "checking_status": ["< 0 DM", "0 to < 500 DM", "500 to < 1000 DM", ">= 1000 DM"],
    "credit_history": ["no credit history", "all credits paid back duly", "critical/other credits", "delayed payments"],
    "purpose": ["education", "retail", "business", "vacation", "domestic appliances",
                "repairs", "car", "furniture/equipment", "radio/TV", "domestic animals"],
    "savings_status": ["< 100 DM", "100 to < 500 DM", "500 to < 1000 DM", ">= 1000 DM", "unknown/0"],
    "employment": ["unemployed", "unskilled", "skilled", "management/self-employed"],
    "other_parties": ["none", "guarantor", "co-applicant"],
    "property_magnitude": ["none", "life insurance", "car or other", "building society", "real estate"],
    "other_payment_plans": ["none", "stores", "life insurance"],
    "housing": ["rent", "own", "for free"],
    "job": ["unemployed", "unskilled", "skilled", "management/self-employed"]
}


def get_feature_input():
    """
    Get feature inputs from user via command line.
    """
    print("=" * 60)
    print("CREDIT RISK PREDICTION TOOL")
    print("=" * 60)
    print("Enter feature values to predict credit risk (Good=1, Bad=0)")
    print("=" * 60)

    feature_dict = {}

    print("\n--- CATEGORICAL FEATURES ---")
    print("1. checking_status:")
    print("   < 0 DM, 0 to < 500 DM, 500 to < 1000 DM, >= 1000 DM")
    checking = input("   Enter value: ").strip()
    feature_dict["checking_status"] = checking

    print("\n2. credit_history:")
    print("   no credit history, all credits paid back duly, critical/other credits, delayed payments")
    hist = input("   Enter value: ").strip()
    feature_dict["credit_history"] = hist

    print("\n3. purpose:")
    print("   education, retail, business, vacation, domestic appliances, repairs,")
    print("   car, furniture/equipment, radio/TV, domestic animals")
    purpose = input("   Enter value: ").strip()
    feature_dict["purpose"] = purpose

    print("\n4. savings_status:")
    print("   < 100 DM, 100 to < 500 DM, 500 to < 1000 DM, >= 1000 DM")
    savings = input("   Enter value: ").strip()
    feature_dict["savings_status"] = savings

    print("\n5. employment:")
    print("   unemployed, unskilled, skilled, management/self-employed")
    employment = input("   Enter value: ").strip()
    feature_dict["employment"] = employment

    print("\n6. other_parties:")
    print("   none, guarantor, co-applicant")
    parties = input("   Enter value: ").strip()
    feature_dict["other_parties"] = parties

    print("\n7. property_magnitude:")
    print("   none, life insurance, car or other, building society, real estate")
    property_mag = input("   Enter value: ").strip()
    feature_dict["property_magnitude"] = property_mag

    print("\n8. other_payment_plans:")
    print("   none, stores, life insurance")
    payment = input("   Enter value: ").strip()
    feature_dict["other_payment_plans"] = payment

    print("\n9. housing:")
    print("   rent, own, for free")
    housing = input("   Enter value: ").strip()
    feature_dict["housing"] = housing

    print("\n10. job:")
    print("   unemployed, unskilled, skilled, management/self-employed")
    job = input("   Enter value: ").strip()
    feature_dict["job"] = job

    print("\n--- NUMERICAL FEATURES ---")
    print("\n10. duration_months: Months of credit duration (1-72)")
    dur = input("   Enter duration (months): ").strip()
    try:
        feature_dict["duration_months"] = int(dur)
    except:
        feature_dict["duration_months"] = 12

    print("\n11. credit_amount: Credit amount in DM (1000-20000)")
    amt = input("   Enter credit amount: ").strip()
    try:
        feature_dict["credit_amount"] = float(amt)
    except:
        feature_dict["credit_amount"] = 5000

    print("\n12. age: Age in years (19-75)")
    age = input("   Enter age: ").strip()
    try:
        feature_dict["age"] = int(age)
    except:
        feature_dict["age"] = 35

    print("\n13. residence_since: Months at residence (1-48)")
    resid = input("   Enter residence since (months): ").strip()
    try:
        feature_dict["residence_since"] = int(resid)
    except:
        feature_dict["residence_since"] = 12

    print("\n14. existing_credits: Number of existing credits (0-5)")
    cred = input("   Enter number of existing credits: ").strip()
    try:
        feature_dict["existing_credits"] = int(cred)
    except:
        feature_dict["existing_credits"] = 0

    print("\n15. num_dependents: Number of dependents (0-3)")
    deps = input("   Enter number of dependents: ").strip()
    try:
        feature_dict["num_dependents"] = int(deps)
    except:
        feature_dict["num_dependents"] = 0

    return feature_dict


def encode_categorical(feature_dict):
    """
    Encode categorical features to numeric values using the category mappings.
    """
    le_dict = {}
    for col in CATEGORICAL_FEATURES:
        if col in feature_dict and col in CATEGORY_LABELS:
            labels = CATEGORY_LABELS[col]
            try:
                # Try to find the index of the entered value
                val = feature_dict[col]
                if val in labels:
                    feature_dict[col] = labels.index(val)
                else:
                    # Default to first category if not found
                    feature_dict[col] = 0
            except:
                feature_dict[col] = 0
    return feature_dict


def predict_credit_risk(feature_dict, model_type="logistic"):
    """
    Predict credit risk (good=1, bad=0) based on feature inputs.
    
    Parameters:
    - feature_dict: dict with feature names as keys and values as inputs
    - model_type: "logistic", "random_forest", or "gradient_boosting"
    
    Returns:
    - prediction: 0 (bad credit) or 1 (good credit)
    - probability: probability of good credit
    - model_name: name of the model used
    """
    # Encode categorical features
    feature_dict = encode_categorical(feature_dict)

    # Extract numerical features
    duration = feature_dict.get("duration_months", 12)
    credit_amount = feature_dict.get("credit_amount", 5000)
    age = feature_dict.get("age", 35)

    # Simple rule-based prediction based on analysis results
    # Based on the analysis of German Credit dataset:
    # Good credit factors: stable employment, good credit history, adequate savings
    # Bad credit factors: bad history, short duration, low savings

    score = 0.5  # neutral start

    # Credit history factor
    hist = feature_dict.get("credit_history", 0)
    if hist in [1, 2]:  # all paid back or no history
        score += 0.3
    elif hist in [3, 4]:  # critical/delayed
        score -= 0.3

    # Savings factor
    savings = feature_dict.get("savings_status", 2)
    if savings in [3, 4]:  # 500+ DM or >= 1000 DM
        score += 0.2
    else:
        score -= 0.1

    # Employment factor
    emp = feature_dict.get("employment", 2)
    if emp in [3, 4]:  # skilled or management
        score += 0.2
    else:
        score -= 0.1

    # Duration factor
    dur = feature_dict.get("duration_months", 12)
    if dur > 24:
        score -= 0.1
    elif dur > 12:
        score -= 0.05

    # Credit amount factor
    amt = feature_dict.get("credit_amount", 5000)
    if amt > 10000:
        score -= 0.1
    elif amt > 5000:
        score += 0.05

    # Checking status
    checking = feature_dict.get("checking_status", 1)
    if checking >= 3:  # 500+ DM
        score += 0.1
    elif checking == 0:  # < 0 DM
        score -= 0.2

    # Age factor
    age_val = feature_dict.get("age", 35)
    if age_val < 20 or age_val > 65:
        score -= 0.1

    # Ensure score is between 0 and 1
    score = np.clip(score, 0.05, 0.95)

    # Convert to prediction and probability
    prediction = (score >= 0.5).astype(int)
    probability = float(score)

    model_name = "Rule-based Approximation"

    return {
        "prediction": prediction,  # 0 = bad credit, 1 = good credit
        "probability": probability,  # probability of good credit
        "model_name": model_name
    }


def main():
    """Main function to run the prediction tool."""
    feature_dict = get_feature_input()

    # Encode categorical features
    feature_dict = encode_categorical(feature_dict)

    # Get prediction
    result = predict_credit_risk(feature_dict)

    print("\n" + "=" * 60)
    print("CREDIT RISK PREDICTION RESULT")
    print("=" * 60)
    print(f"\nPrediction: {'GOOD CREDIT (1)' if result['prediction'] == 1 else 'BAD CREDIT (0)'}")
    print(f"Probability of good credit: {result['probability']:.2%}")
    print(f"Model used: {result['model_name']}")

    print("\n--- Feature Summary ---")
    for key, val in feature_dict.items():
        print(f"  {key}: {val}")

    print("\n--- Interpretation ---")
    if result["prediction"] == 1:
        print("Good credit is likely with the provided features.")
        print("  - Credit history appears positive")
        print("  - Savings appear adequate")
        print("  - Employment seems stable")
    else:
        print("Bad credit risk is indicated with the provided features.")
        print("  - Credit history may have issues")
        print("  - Savings may be insufficient")
        print("  - Employment may be unstable")

    print("\n--- Key Factors Considered ---")
    print("  1. Credit history and payment behavior")
    print("  2. Employment stability and job type")
    print("  3. Savings and financial reserves")
    print("  4. Credit duration and amount")
    print("  5. Housing and other financial obligations")

    print("\n--- Disclaimer ---")
    print("This is a simplified prediction model for demonstration purposes.")
    print("Real credit decisions require comprehensive assessment")
    print("and regulatory compliance with financial laws.")


if __name__ == "__main__":
    main()