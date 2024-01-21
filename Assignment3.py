import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from matplotlib.ticker import FuncFormatter

df_countries=pd.read_csv('E:\ZK03_P2\API_AG.LND.AGRI.K2_DS2_en_csv_v2_6304041.csv',skiprows=4)

def agriculture_land_prediction(df_countries, countries):
    indicator_name = 'Agricultural land (sq. km)'

    def agriculture_land_model(year, a, b, c):
        return a * np.exp(b * (year - 1990)) + c

    def plot_country_agriculture_land(country_data, country_name):
        # Extract years and agriculture land data for the specified country and indicator
        years = country_data.columns[4:]  # Assuming the years start from the 5th column
        agriculture_land = country_data.iloc[:, 4:].values.flatten()

        # Convert years to numeric values
        years_numeric = pd.to_numeric(years, errors='coerce')
        agriculture_land = pd.to_numeric(agriculture_land, errors='coerce')

        # Remove rows with NaN or inf values
        valid_data_mask = np.isfinite(years_numeric) & np.isfinite(agriculture_land)
        years_numeric = years_numeric[valid_data_mask]
        agriculture_land = agriculture_land[valid_data_mask]

        # Curve fitting with increased maxfev
        params, covariance = curve_fit(agriculture_land_model, years_numeric, agriculture_land, p0=[1, -0.1, 90], maxfev=10000)

        # Optimal parameters
        a_opt, b_opt, c_opt = params

        # Generate model predictions for the year 2040
        year_2040 = 2040
        agriculture_land_2040 = agriculture_land_model(year_2040, a_opt, b_opt, c_opt)

        # Plot the original data and the fitted curve
        plt.figure(figsize=(10, 6))
        plt.scatter(years_numeric, agriculture_land, label='Original Data', color='blue', alpha=0.7, edgecolors='black')
        plt.plot(years_numeric, agriculture_land_model(years_numeric, a_opt, b_opt, c_opt), label='Fitted Curve', color='red')

        # Highlight the prediction for 2040
        plt.scatter(year_2040, agriculture_land_2040, color='green', marker='*', label='Prediction for 2040', s=100, edgecolors='black')

        # Add labels and legend
        plt.title(f'Agriculture Land Prediction for {country_name}')
        plt.xlabel('Year')
        plt.ylabel('Agriculture Land (sq. km)')
        plt.legend()
        plt.grid(True)

        # Show the plot
        plt.show()

    for country_name in countries:
        country_data = df_countries[(df_countries['Country Name'] == country_name) & (df_countries['Indicator Name'] == indicator_name)]
        if not country_data.empty:
            plot_country_agriculture_land(country_data, country_name)

# Example usage
countries_to_plot = ['Argentina', 'Australia', 'Pakistan']
agriculture_land_prediction(df_countries, countries_to_plot)

# Extract data for the years 1970 and 2020
years = ['1970', '2020']
agriculture_land_data = df_countries[['Country Name'] + years]

# Drop rows with missing values
agriculture_land_data = agriculture_land_data.dropna()

# Set 'Country Name' as the index
agriculture_land_data.set_index('Country Name', inplace=True)

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(agriculture_land_data)

# Perform KMeans clustering
n_clusters = 2
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
labels = kmeans.fit_predict(normalized_data)

# Add cluster labels to the DataFrame
agriculture_land_data['Cluster'] = labels

# Define a custom formatter to display numbers in thousands
def format_thousands(x, pos):
    return f'{int(x / 1000)}K'

# Custom color map for clusters
colors = np.array(['red', 'blue'])

# Visualize the clusters
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Cluster for 1970
axs[0].scatter(agriculture_land_data[years[0]], agriculture_land_data.index, c=colors[labels], cmap='viridis')
axs[0].set_title(f'Agriculture land in {years[0]}')
axs[0].set_xlabel('Agriculture Land')
axs[0].set_ylabel('Countries')
axs[0].xaxis.set_major_formatter(FuncFormatter(format_thousands))  # Apply custom formatter to x-axis

# Cluster for 2020
axs[1].scatter(agriculture_land_data[years[1]], agriculture_land_data.index, c=colors[labels], cmap='viridis')
axs[1].set_title(f'Agriculture land in {years[1]}')
axs[1].set_xlabel('Agriculture Land')
axs[1].set_ylabel('Countries')
axs[1].xaxis.set_major_formatter(FuncFormatter(format_thousands))  # Apply custom formatter to x-axis

# Manually set y-axis label
for ax in axs:
    ax.set_yticks([])
    ax.set_yticklabels([])

plt.tight_layout()
plt.show()

years = ['1970', '2020']
agriculture_land_data = df_countries[['Country Name'] + years]

agriculture_land_data.head()

# Drop rows with missing values
agriculture_land_data = agriculture_land_data.dropna()

