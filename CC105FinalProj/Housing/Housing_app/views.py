from django.shortcuts import render, redirect
import pandas as pd
import numpy as np
import pickle
import os
import joblib
from pathlib import Path
from .models import HousingPrediction
from django.contrib.auth import login, authenticate, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Get the base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load model and scaler
def load_model():
    try:
        # Try to load from DataModels directory
        model_path = os.path.join(BASE_DIR, 'DataModels', 'housing_price_model.pkl')
        scaler_path = os.path.join(BASE_DIR, 'DataModels', 'housing_scaler.pkl')
        
        model = pickle.load(open(model_path, 'rb'))
        scaler = pickle.load(open(scaler_path, 'rb'))
        
        return model, scaler
    except Exception as e:
        # Fallback to parent directory
        try:
            model_path = os.path.join(BASE_DIR, 'housing_price_model.pkl')
            scaler_path = os.path.join(BASE_DIR, 'housing_scaler.pkl')
            
            model = pickle.load(open(model_path, 'rb'))
            scaler = pickle.load(open(scaler_path, 'rb'))
            
            return model, scaler
        except Exception as inner_e:
            print(f"Error loading model: {str(inner_e)}")
            return None, None

# Home page - project intro
def home(request):
    return render(request, 'housing_app/home.html')

# Login view
def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f"You are now logged in as {username}")
                return redirect('index')
            else:
                messages.error(request, "Invalid username or password.")
        else:
            messages.error(request, "Invalid username or password.")
    form = AuthenticationForm()
    return render(request, 'housing_app/login.html', {'form': form})

# Register view
def register_view(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Registration successful!")
            return redirect('index')
        else:
            messages.error(request, "Registration failed. Please correct the errors.")
    else:
        form = UserCreationForm()
    return render(request, 'housing_app/register.html', {'form': form})

# Logout view
def logout_view(request):
    logout(request)
    messages.info(request, "You have successfully logged out.")
    return redirect('home')

# Dashboard view - statistics and charts
@login_required
def dashboard(request):
    # Get all predictions from the database
    predictions = HousingPrediction.objects.all().order_by('-created_at')
    
    # If no predictions, show empty dashboard
    if not predictions:
        return render(request, 'housing_app/dashboard.html', {
            'num_predictions': 0,
            'avg_price': 0,
            'min_price': 0,
            'max_price': 0,
            'no_data': True,
            'predictions': []
        })
    
    # Calculate statistics
    prices = [p.predicted_price for p in predictions]
    avg_price = sum(prices) / len(prices)
    min_price = min(prices)
    max_price = max(prices)
    
    # Create chart for neighborhood distribution
    neighborhood_counts = {}
    for p in predictions:
        if p.neighborhood in neighborhood_counts:
            neighborhood_counts[p.neighborhood] += 1
        else:
            neighborhood_counts[p.neighborhood] = 1
    
    # Find the most predicted neighborhood
    most_predicted_neighborhood = max(neighborhood_counts, key=neighborhood_counts.get)
    
    # Create green to black color gradient for charts
    green_color = '#1E8449'  # Primary green
    black_color = '#212121'  # Black
    
    # Create a gradient colormap from green to black
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list('green_black', [green_color, black_color])
    
    # Create bar chart with gradient colors
    plt.figure(figsize=(10, 6))
    bars = plt.bar(neighborhood_counts.keys(), neighborhood_counts.values())
    
    # Apply gradient colors to bars
    num_bars = len(bars)
    for i, bar in enumerate(bars):
        # Calculate color position in gradient based on bar position
        color_val = i / max(1, num_bars - 1)
        bar.set_color(cmap(color_val))
    
    plt.title('Number of Predictions by Neighborhood')
    plt.xlabel('Neighborhood')
    plt.ylabel('Count')
    plt.tight_layout()
    
    # Save chart to string buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    neighborhood_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Create pie chart for condition distribution
    condition_counts = {}
    for p in predictions:
        if p.condition in condition_counts:
            condition_counts[p.condition] += 1
        else:
            condition_counts[p.condition] = 1
    
    # Create pie chart with gradient colors
    plt.figure(figsize=(8, 8))
    
    # Generate colors from the gradient for each pie slice
    pie_colors = [cmap(i / max(1, len(condition_counts) - 1)) for i in range(len(condition_counts))]
    
    plt.pie(condition_counts.values(), labels=condition_counts.keys(), autopct='%1.1f%%', colors=pie_colors)
    plt.title('Proportion of Property Conditions')
    plt.tight_layout()
    
    # Save chart to string buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    condition_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    # Get model performance metrics
    # This would normally come from model evaluation, but we'll use placeholder values for now
    model_accuracy = 0.92  # 92% accuracy
    model_accuracy_percent = model_accuracy * 100  # Convert to percentage for the template
    model_mse = 1250000  # Mean Squared Error
    
    # Return the dashboard view with statistics and charts
    return render(request, 'housing_app/dashboard.html', {
        'num_predictions': len(predictions),
        'avg_price': avg_price,
        'min_price': min_price,
        'max_price': max_price,
        'neighborhood_chart': neighborhood_chart,
        'condition_chart': condition_chart,
        'model_accuracy': model_accuracy,
        'model_accuracy_percent': model_accuracy_percent,
        'model_mse': model_mse,
        'predictions': predictions,
        'most_predicted_neighborhood': most_predicted_neighborhood,
        'no_data': False
    })

# Prediction form - requires login
@login_required
def index(request):
    return render(request, 'housing_app/index.html')

# Process form data and make prediction - requires login
@login_required
def predict(request):
    if request.method == 'POST':
        # Get form data
        sqft = float(request.POST.get('sqft'))
        bedrooms = int(request.POST.get('bedrooms'))
        bathrooms = float(request.POST.get('bathrooms'))
        lot_size = float(request.POST.get('lot_size'))
        year_built = int(request.POST.get('year_built'))
        stories = int(request.POST.get('stories'))
        neighborhood = request.POST.get('neighborhood')
        garage = int(request.POST.get('garage'))
        condition = request.POST.get('condition')
        
        # Store the input data for display on result page
        input_data = {
            'sqft': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'lot_size': lot_size,
            'year_built': year_built,
            'stories': stories,
            'neighborhood': neighborhood,
            'garage': garage,
            'condition': condition
        }
        
        # Load the model and scaler
        model, scaler = load_model()
        
        if model is None or scaler is None:
            return render(request, 'housing_app/index.html', {
                'prediction': 0,
                'error': 'Error loading model. Please try again later.'
            })
        
        # Prepare the input features (similar to what we did in the model script)
        # Create a DataFrame with the input data
        input_df = pd.DataFrame([{
            'sqft': sqft,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'lot_size': lot_size,
            'year_built': year_built,
            'stories': stories,
            'garage': garage
        }])
        
        # Create a new dataframe with the exact same columns that were used during training
        # This ensures column order and names match exactly what the model expects
        
        # Example neighborhood encoding - match exactly what was in training set
        # Add all possible neighborhood values from the training data
        neighborhood_cols = ['neighborhood_Downtown', 'neighborhood_Lakefront', 'neighborhood_Suburbs']
        for col in neighborhood_cols:
            input_df[col] = 0  # Initialize all to 0
            
        # Set the appropriate neighborhood to 1
        if neighborhood == 'Downtown':
            input_df['neighborhood_Downtown'] = 1
        elif neighborhood == 'Lakefront':
            input_df['neighborhood_Lakefront'] = 1
        elif neighborhood == 'Suburbs':
            input_df['neighborhood_Suburbs'] = 1
            
        # Example condition encoding - match exactly what was in training set
        # Add all possible condition values from the training data
        condition_cols = ['condition_Excellent', 'condition_Fair', 'condition_Good']
        for col in condition_cols:
            input_df[col] = 0  # Initialize all to 0
            
        # Set the appropriate condition to 1
        if condition == 'Excellent':
            input_df['condition_Excellent'] = 1
        elif condition == 'Fair':
            input_df['condition_Fair'] = 1
        elif condition == 'Good':
            input_df['condition_Good'] = 1
        
        # Ensure columns are in the same order as during training
        expected_columns = ['sqft', 'bedrooms', 'bathrooms', 'lot_size', 'year_built', 
                           'stories', 'garage', 'neighborhood_Downtown', 
                           'neighborhood_Lakefront', 'neighborhood_Suburbs',
                           'condition_Excellent', 'condition_Fair', 'condition_Good']
        
        # Get the feature names expected by the scaler
        scaler_features = scaler.feature_names_in_ if hasattr(scaler, 'feature_names_in_') else expected_columns
        
        # Make sure the input dataframe has exactly the same columns in the same order
        input_df = input_df.reindex(columns=scaler_features, fill_value=0)
        
        # Scale the features using the pre-trained scaler
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        
        # Round to nearest thousand for a cleaner display
        prediction = round(prediction / 1000) * 1000
        
        # Save prediction to database
        HousingPrediction.objects.create(
            sqft=sqft,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            lot_size=lot_size,
            year_built=year_built,
            stories=stories,
            neighborhood=neighborhood,
            garage=garage,
            condition=condition,
            predicted_price=prediction
        )
        
        # Return the result page
        return render(request, 'housing_app/prediction_result.html', {
            'prediction': prediction,
            'input_data': input_data
        })
    
    # If not POST, redirect to the form
    return render(request, 'housing_app/index.html')
