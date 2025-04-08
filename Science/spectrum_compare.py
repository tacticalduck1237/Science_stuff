import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
# Load the CSV file
def load_spectra(file_path):
    df = pd.read_csv(file_path)
    spectrum1 = df.iloc[:, 0].values  # First column
    spectrum2 = df.iloc[:, 1].values  # Second column
    return spectrum1, spectrum2

# Calculate Correlation Coefficient
def calculate_correlation(spectrum1, spectrum2):
    return np.corrcoef(spectrum1, spectrum2)[0, 1]

# Calculate Mean Squared Error (MSE)
def calculate_mse(spectrum1, spectrum2):
    return np.mean((spectrum1 - spectrum2) ** 2)

# Calculate Root Mean Squared Error (RMSE)
def calculate_rmse(spectrum1, spectrum2):
    return np.sqrt(calculate_mse(spectrum1, spectrum2))

# Calculate Spectral Angle Mapper (SAM)
def calculate_sam(spectrum1, spectrum2):
    # Convert spectra to vectors and calculate the angle between them
    dot_product = np.dot(spectrum1, spectrum2)
    norm1 = np.linalg.norm(spectrum1)
    norm2 = np.linalg.norm(spectrum2)
    cos_theta = dot_product / (norm1 * norm2)
    return np.arccos(cos_theta)  # Angle in radians

# Main function
def main(file_path):
    # Load spectra
    spectrum1, spectrum2 = load_spectra(file_path)
    
    # Calculate metrics
    correlation = calculate_correlation(spectrum1, spectrum2)
    mse = calculate_mse(spectrum1, spectrum2)
    rmse = calculate_rmse(spectrum1, spectrum2)
    sam = calculate_sam(spectrum1, spectrum2)
    
    # Output results
    print(f"Correlation Coefficient: {correlation:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Spectral Angle Mapper (SAM) in radians: {sam:.4f}")

# Example usage
if __name__ == "__main__":
    file_path = "/home/doomer/Downloads/MSU_MFK_AI/Science/1.csv"  # Replace with your CSV file path
    main(file_path)