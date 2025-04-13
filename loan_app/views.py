from django.shortcuts import render
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import os

# Load model & scaler only once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'ml_model/loan_ann_model.h5'))
scaler = joblib.load(os.path.join(BASE_DIR, 'ml_model/scaler.pkl'))

def index(request):
    return render(request, 'loan_app/index.html')

def predict(request):
    prediction = None

    if request.method == 'POST':
        try:
            data = [
                float(request.POST.get('annual_income')),
                float(request.POST.get('credit_score')),
                float(request.POST.get('loan_amount')),
                float(request.POST.get('loan_tenure')),
                float(request.POST.get('monthly_expenses')),
                float(request.POST.get('age')),
                float(request.POST.get('existing_loans')),
                int(request.POST.get('employment_employed')),
                int(request.POST.get('employment_self')),
                int(request.POST.get('residence_owned')),
                int(request.POST.get('residence_rented')),
                int(request.POST.get('purpose_edu')),
                int(request.POST.get('purpose_home')),
                int(request.POST.get('purpose_medical')),
                int(request.POST.get('purpose_wedding')),
                int(request.POST.get('married')),
                int(request.POST.get('single'))
            ]

            input_scaled = scaler.transform([data])
            result = model.predict(input_scaled)[0][0]
            prediction = 1 if result >= 0.5 else 0

        except Exception as e:
            print("Prediction error:", e)
            prediction = None

    return render(request, 'loan_app/index.html', {'prediction': prediction})

def insights(request):
    return render(request, 'loan_app/insights.html')
