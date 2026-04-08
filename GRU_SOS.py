from datetime import datetime
import time
import warnings
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xarray as xr

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# 1. 数据加载
climate_path = ''
soil_texture_path = ''
sos_path = ''

# Load climate data
temp_data_xr = load data**
temp_data = load data**
rain_data = load data**
swrd_data = load data**
vpd_data = load data**
sm_data = load data**

# Load land cover and soil data
IGBP_data = load data**
soil_clay_data = load data**
soil_sand_data = load data**
soil_oc_data = xr.open_dataset(soil_texture_path + 'T_OC_0.25deg.nc').sel(
    lat=slice(30, 90))['T_OC'].values

# Load phenology data (LOD)
LOD_data_xr = xr.open_dataset(sos_path + r'2001_2021_Vegetation_SOS_EOS_EVI.nc').sel(
    lat=slice(30, 90), time=slice('2001-01-01', '2018-12-31'))['SOS']
LOD_data = LOD_data_xr.values


print(f"Climate data shapes: temp={temp_data.shape}, rain={rain_data.shape}, "
      f"swrd={swrd_data.shape}, vpd={vpd_data.shape}, sm={sm_data.shape}")
print(f"Land cover shape: IGBP={IGBP_data.shape}")
print(f"Phenology data shape: LOD={LOD_data.shape}")
print(f"Soil data shapes: clay={soil_clay_data.shape}, sand={soil_sand_data.shape}, "
      f"oc={soil_oc_data.shape}")

mask = (
    (IGBP_data >= 1) & (IGBP_data <= 4) &
    (np.all(LOD_data > 10, axis=0))
)
lat_indices, lon_indices = np.where(mask)
print(f"Valid pixels after masking: {len(lat_indices)}")

# Define the prediction years based on LOD data
prediction_years =  project timeeriers # (2021, 2060)

# 3. 数据预处理和序列构建
print("\nStarting data preprocessing and sequence construction...")

X_data = []  # List to store input sequences (daily climate data)
X_static = []  # List to store static features (soil properties, IGBP)
locations = []  # Store (lat_idx, lon_idx) for each sample

# Define the daily climate features
climate_features = [temp_data, rain_data, swrd_data, vpd_data, sm_data]
num_features = len(climate_features)

# Define static features
static_features = [soil_clay_data, soil_sand_data, soil_oc_data, IGBP_data]
num_static_features = len(static_features)
daily_timesteps = 365  # Number of days in the sequence for each year

for year_idx, target_year in enumerate(prediction_years):
    # Calculate the year index in the CMIP6 data
    # Assuming CMIP6 data starts from 2021
    year_offset = year_idx * 365  # Offset in days for each year

    for i, (lat_idx, lon_idx) in enumerate(zip(lat_indices, lon_indices)):
        # Extract daily climate data for the current year, location, and timesteps
        daily_climate_sequence = np.zeros((daily_timesteps, num_features))
        for feature_idx, feature_data in enumerate(climate_features):
            start_idx = year_offset
            end_idx = year_offset + daily_timesteps
            if end_idx <= len(feature_data):
                daily_climate_sequence[:, feature_idx] = feature_data[start_idx:end_idx, lat_idx, lon_idx]
            else:
                print(f"Warning: Not enough data for year {target_year}")
                continue

        # Extract static features for the current location
        static_sequence = np.zeros(num_static_features)
        for static_idx, static_data in enumerate(static_features):
            static_sequence[static_idx] = static_data[lat_idx, lon_idx]

        # Check for NaN values in the sequence and skip if present
        if np.isnan(daily_climate_sequence).any() or np.isnan(static_sequence).any():
            continue

        X_data.append(daily_climate_sequence)
        X_static.append(static_sequence)
        locations.append((lat_idx, lon_idx, target_year))  # Store location and year for tracking

print(f"Total samples after preprocessing: {len(X_data)}")

# Convert to numpy arrays
X_data = np.array(X_data, dtype=np.float32)
X_static = np.array(X_static, dtype=np.float32)


# 4. model 
class GRUPhenologyPredictor(nn.Module):
    def __init__(self, input_size, static_size, hidden_size, num_layers, output_size, dropout_rate=0.2):
        super(GRUPhenologyPredictor, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # Batch normalization for input features
        self.input_bn = nn.BatchNorm1d(input_size)
        self.static_bn = nn.BatchNorm1d(static_size)

        # GRU layer with dropout
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)

        # Static features processing with dropout
        self.static_fc = nn.Sequential(
            nn.Linear(static_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Additional layers for better feature extraction
        self.combined_fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # Final output layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, x_static):
        batch_size = x.size(0)
        seq_len = x.size(1)

        # Apply batch normalization to input features
        # Reshape for BatchNorm1d
        x = x.view(-1, x.size(-1))
        x = self.input_bn(x)
        x = x.view(batch_size, seq_len, -1)

        # Apply batch normalization to static features
        x_static = self.static_bn(x_static)

        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        # Forward propagate GRU
        out, _ = self.gru(x, h0)

        # Get last timestep output
        gru_output = out[:, -1, :]

        # Process static features
        static_output = self.static_fc(x_static)

        # Concatenate GRU and static outputs
        combined_output = torch.cat([gru_output, static_output], dim=1)

        # Process combined features
        combined_output = self.combined_fc(combined_output)

        # Final prediction
        out = self.fc(combined_output)
        return out


# 5. load model
# Model hyperparameters (must match the trained model)
input_size = num_features
static_size = num_static_features
hidden_size = 128
num_layers = 3
output_size = 1
dropout_rate = 0.3

# Initialize model
model = GRUPhenologyPredictor(input_size, static_size, hidden_size, num_layers, output_size, dropout_rate)

# Load the trained model weights
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load(r'best_gru_phenology_signal_model_NDVI.pth',
                        map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# 6. data stander
# Load normalization parameters from the training data
train_data = np.load(r'train_normalization_params.npz')
X_mean = train_data['X_mean']
X_std = train_data['X_std']
X_static_mean = train_data['X_static_mean']
X_static_std = train_data['X_static_std']
y_mean = train_data['y_mean']
y_std = train_data['y_std']

# Normalize the prediction data
X_normalized = (X_data - X_mean) / (X_std + 1e-8)
X_static_normalized = (X_static - X_static_mean) / (X_static_std + 1e-8)


# 7. create data load 
class PredictionDataset(Dataset):
    def __init__(self, X, X_static):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.X_static[idx]


pred_dataset = PredictionDataset(X_normalized, X_static_normalized)
pred_loader = DataLoader(pred_dataset, batch_size=512, shuffle=False)

# 8. projection
print("Making predictions...")
predictions = []

with torch.no_grad():
    for inputs, inputs_static in pred_loader:
        inputs, inputs_static = inputs.to(device), inputs_static.to(device)
        outputs = model(inputs, inputs_static)
        predictions.extend(outputs.cpu().numpy().flatten())

# Denormalize predictions
predictions = np.array(predictions) * y_std + y_mean

# 9. 
future_lsd = np.full((len(prediction_years), 240, 1440), np.nan)
for i, (lat_idx, lon_idx, year) in enumerate(locations):
    year_idx = prediction_years.tolist().index(year)
    future_lsd[year_idx, lat_idx, lon_idx] = predictions[i]

# 10. save data
# Create xarray DataArray with proper coordinates
lsd_da = xr.DataArray(
    future_lsd,
    coords={
        'time': prediction_years,
        'lat': temp_data_xr.coords['lat'].values,
        'lon': temp_data_xr.coords['lon'].values
    },
    dims=['time', 'lat', 'lon'],
    name='LOD'
)

# Save to netCDF file
lsd_da.to_netcdf(r'path')



