import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Load data and drop last (transmission) point:
with open('SERAPH_R118_absorbtion (1).json') as f:
    data = json.load(f)['datasetColl'][0]['data'][:-1]
wavelengths = np.array([p['value'][0] for p in data])
transmissions = np.array([p['value'][1] for p in data])

plt.figure(figsize=(10, 6))
plt.scatter(wavelengths, transmissions, color='orange', label='Original Data')
colors = ['blue', 'green', 'red', 'purple', 'brown', 'cyan', 'magenta']
for i, degree in enumerate(range(12, 19, 2)):
    coeffs = np.polyfit(wavelengths, transmissions, degree)
    poly = np.poly1d(coeffs)
    fitted = poly(wavelengths)
    plt.plot(
        wavelengths,
        fitted,
        color=colors[i % len(colors)],
        linestyle='-',
        label=f'{degree}th-Degree Fit'
    )

plt.title('Transmission vs Wavelength (Polynomial Fits, Degree 12–19, step 2)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('poly12_19_trans_fits.png')
plt.show()

# Fit 14th-degree polynomial
degree = 14
coeffs_14 = np.polyfit(wavelengths, transmissions, degree)
poly_14 = np.poly1d(coeffs_14)
fitted_14 = poly_14(wavelengths)
residues_14 = transmissions - fitted_14

# Plot 14th-degree fit
plt.figure(figsize=(10, 6))
plt.scatter(wavelengths, transmissions, color='orange', label='Original Data')
plt.plot(wavelengths, fitted_14, color='blue', label='14th-Degree Fit')
plt.title('Transmission vs Wavelength (14th-Degree Polynomial Fit)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('poly14_trans_fit.png')
plt.show()

# Plot residues
plt.figure(figsize=(10, 4))
plt.scatter(wavelengths, residues_14, color='red', label='Residues')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residues of 14th-Degree Polynomial Fit')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Residue')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('poly14_trans_residues.png')
plt.show()

# Print parameters table and equation
print("14th-degree polynomial coefficients (highest degree first):")
for i, c in enumerate(coeffs_14):
    print(f"a{i:02d} (deg {degree-i}): {c:.6e}")
print("\nPolynomial equation:")
terms = [f"{c:.3e}*x^{degree-i}" for i, c in enumerate(coeffs_14[:-1])]
terms.append(f"{coeffs_14[-1]:.3e}")
equation = " + ".join(terms)
print("y =", equation)

# Save coefficients and equation to JSON
output = {
    "degree": degree,
    "coefficients": coeffs_14.tolist(),
    "equation": "y = " + equation
}
with open('poly14_trans_fit.json', 'w') as f:
    json.dump(output, f, indent=2)

# --- Sine fit to residues ---
# Define sine function for fitting versus wavelength
def sine_model(x, A, T, phi, offset):
    return A * np.sin(2 * np.pi * x / T + phi) + offset

# Compute residues of 14th-degree polynomial fit
residues_14 = transmissions - poly_14(wavelengths)

# Improved initial guesses:
A0 = (residues_14.max() - residues_14.min()) / 2
T0 = wavelengths.max() - wavelengths.min()  # assume one period spans full range
phi0 = 0.0
offset0 = np.mean(residues_14)
p0 = [A0, T0, phi0, offset0]

# Perform nonlinear least-squares fit with higher maxfev
popt, pcov = curve_fit(
    sine_model,
    wavelengths,
    residues_14,
    p0=p0,
    maxfev=10000
)
A_fit, T_fit, phi_fit, offset_fit = popt

# Compute fitted sine values over the wavelength range
# Plot residues for polynomial fits from degree 5 to 20
plt.figure(figsize=(12, 6))
for degree in range(5, 21):
    coeffs = np.polyfit(wavelengths, transmissions, degree)
    poly = np.poly1d(coeffs)
    residues = transmissions - poly(wavelengths)
    plt.plot(wavelengths, residues, label=f'Degree {degree}')
plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
plt.title('Residues of Polynomial Fits (Degrees 5 to 20)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Residue')
plt.legend(ncol=2)
plt.grid(True)
plt.tight_layout()
plt.savefig('poly_trans_residues_5_20.png')
plt.show()

# --- Additional visualization: Heatmap of residues vs degree and wavelength ---
# Compute residues for each degree and store in a 2D array
degrees = np.arange(5, 21)
residues_matrix = np.zeros((len(degrees), len(wavelengths)))
for i, degree in enumerate(degrees):
    coeffs = np.polyfit(wavelengths, transmissions, degree)
    poly = np.poly1d(coeffs)
    residues_matrix[i, :] = transmissions - poly(wavelengths)

plt.figure(figsize=(12, 6))
im = plt.imshow(
    residues_matrix,
    aspect='auto',
    extent=[wavelengths.min(), wavelengths.max(), degrees[-1]+0.5, degrees[0]-0.5],
    cmap='RdBu',
    vmin=-np.max(np.abs(residues_matrix)),
    vmax=np.max(np.abs(residues_matrix))
)
plt.colorbar(im, label='Residue')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Polynomial Degree')
plt.title('Residue Heatmap: Degree vs Wavelength')
plt.tight_layout()
plt.savefig('residue_trans_heatmap.png')
plt.show()

# --- Plot standard deviation of residues vs polynomial degree ---
stdevs = []
for degree in range(5, 21):
    coeffs = np.polyfit(wavelengths, transmissions, degree)
    poly = np.poly1d(coeffs)
    residues = transmissions - poly(wavelengths)
    stdevs.append(np.std(residues))

plt.figure(figsize=(8, 5))
plt.plot(range(5, 21), stdevs, color='purple', linestyle='-')
plt.vlines(14, ymin=0, ymax=max(stdevs), color='red', linestyle='-', label='14th Degree Fit')
plt.xlabel('Polynomial Degree')
plt.ylabel('Residue Standard Deviation')
plt.title('Standard Deviation of Residues vs Polynomial Degree')
plt.grid(axis='x')
plt.tight_layout()
plt.savefig('residue_trans_stdev_vs_degree.png')
plt.show()

# Adding plot of data without fits
plt.figure(figsize=(10, 6))
plt.scatter(wavelengths, transmissions, color='blue', s=20, alpha=0.6, label='KOSEN Spectrometer Data')
plt.title('Transmission vs Wavelength (KOSEN Spectrometer Data)')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Transmission')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('kosen_spectrometer_transmission_data.png')
plt.show()