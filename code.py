# 1. Import Necessary Libraries
# We begin by importing essential libraries for data manipulation and visualization.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 2. Load and Preprocess Yield Curve Data
# We load the yield curve data, convert the 'DATE' column to datetime, filter the data for the desired date range, set 'DATE' as the index, and sort the data.

# Load and Preprocess Yield Curve Data

# Load the data for yield curve
df_Yield = pd.read_excel('C:/Users/Bruno/OneDrive/Documents/Master FTD/Asset Pricing/Webstat_Export_20241026.xlsx', sheet_name="DATA")


# Convert 'DATE' column to datetime
df_Yield['DATE'] = pd.PeriodIndex(df_Yield['DATE'], freq='Q').to_timestamp(how='end')

# Set 'DATE' as index
df_Yield.set_index('DATE', inplace=True)

# Preview the data
print("Yield Curve Data:")
print(df_Yield.head())
print(df_Yield.tail())

# 3. Load and Preprocess Macroeconomic Data

# Load and Preprocess Macroeconomic Data

# Load GDP data
df_GDP = pd.read_excel('C:/Users/Bruno/OneDrive/Documents/Master FTD/Asset Pricing/econ-gen-pib-composante-trim.xlsx', sheet_name="DATA")
df_GDP['DATE'] = pd.PeriodIndex(df_GDP['DATE'], freq='Q').to_timestamp(how='end')
df_GDP.set_index('DATE', inplace=True)



# Load CPI data
df_CPI = pd.read_excel('C:/Users/Bruno/OneDrive/Documents/Master FTD/Asset Pricing/FRACPIALLQINMEI.xlsx', sheet_name="DATA")
df_CPI['CPI_QoQ_LogChange'] = np.log(df_CPI['CPI'].pct_change() + 1)
df_CPI['DATE'] = pd.PeriodIndex(df_CPI['DATE'], freq='Q').to_timestamp(how='end')
df_CPI.set_index('DATE', inplace=True)


# Load Unemployment data
df_CHO = pd.read_excel('C:/Users/Bruno/OneDrive/Documents/Master FTD/Asset Pricing/Essentiel_Chomage_donnees.xlsx', sheet_name="DATA")
df_CHO['DATE'] = pd.PeriodIndex(df_CHO['DATE'], freq='Q').to_timestamp(how='end')
df_CHO.set_index('DATE', inplace=True)



# 4. Merge Dataframes
# We merge the yield curve data with the macroeconomic data on the 'DATE' index and ensure there are no missing values.

# Merge all dataframes on 'DATE' index
data = df_Yield.join([df_GDP, df_CPI['CPI_QoQ_LogChange'], df_CHO], how='inner')

# Ensure there are no missing values
data.dropna(inplace=True)

data.index = pd.to_datetime(data.index)
duplicates = data.index.duplicated()

# If duplicates are found, remove them (or handle as needed)
if duplicates.any():
    print("Duplicates found. Removing duplicates.")
    data = data[~duplicates]  # Removes duplicated rows based on index


# Preview the merged data
print("Merged Data:")
print(data.head())
print(data.tail())



#%%%%


from scipy.optimize import minimize

# Nelson-Siegel function
def nelson_siegel(maturity, beta1, beta2, beta3, lambda_param):
    term1 = beta1
    term2 = beta2 * ((1 - np.exp(-maturity / lambda_param)) / (maturity / lambda_param))
    term3 = beta3 * (((1 - np.exp(-maturity / lambda_param)) / (maturity / lambda_param)) - np.exp(-maturity / lambda_param))
    return term1 + term2 + term3


# 5.2 Fit the Model for Each Time Point
# We fit the Nelson-Siegel model to each time point, store the beta parameters, and update the initial parameters for better convergence.

# Objective function for optimization
def ns_objective(params, maturities, yields):
    beta1, beta2, beta3, lambda_param = params
    fitted_yields = nelson_siegel(maturities, beta1, beta2, beta3, lambda_param)
    return np.sum((yields - fitted_yields) ** 2)

# Define maturities in years
maturities = np.array([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])

# Initialize dictionary to store parameters
ns_params = {}

# Initialize initial parameters
initial_params = [0.03, -0.02, 0.02, 1.0]  # Initial guess for the first date

# Fit the model for each date
for date, row in df_Yield.iterrows():
    yields = row.values  # Yields for different maturities
    try:
        result = minimize(ns_objective, initial_params, args=(maturities, yields), method='L-BFGS-B')
        ns_params[date] = result.x  # Store the fitted parameters
        initial_params = result.x  # Update initial parameters for the next date
    except Exception as e:
        print(f"Optimization failed for date {date}: {e}")
        continue

# Convert ns_params into a DataFrame
ns_params_df = pd.DataFrame.from_dict(ns_params, orient='index', columns=['beta1', 'beta2', 'beta3', 'lambda'])
ns_params_df.index.name = 'DATE'

# Sort the DataFrame by date
ns_params_df.sort_index(inplace=True)

# Preview the beta parameters
print("Nelson-Siegel Parameters:")
print(ns_params_df.head(20))

# 6. Analyze Beta Parameters
# We visualize the beta parameters over time to observe any trends or patterns.

# Plot beta parameters over time
ns_params_df[['beta1', 'beta2', 'beta3']].plot(subplots=True, figsize=(10, 8))
plt.tight_layout()
plt.show()

# 7. Merge Beta Parameters with Macroeconomic Data
# We merge the beta parameters with the macroeconomic data for further analysis.

# Merge beta parameters with macroeconomic data
merged_data = ns_params_df.join(data, how='inner')

# Ensure there are no missing values
merged_data.dropna(inplace=True)

# Ensure 'DATE' index is a PeriodIndex with quarterly frequency
merged_data.index = pd.PeriodIndex(merged_data.index, freq='Q')

# Verify if the index now shows as a PeriodIndex with quarterly frequency
print(merged_data.index)  # Should now show as PeriodIndex with freq='Q'


# Preview the merged data
print("Merged Beta Parameters and Macroeconomic Data:")
print(merged_data.head())



#%%%

#We can also do a PCA analysis

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt



# Calculate differences to get yield curve change
df_Yield_diff = df_Yield.diff().dropna()

# Perform PCA
pca = PCA(n_components=3)
pca.fit(df_Yield)
factor_loadings = pca.components_.T

# Plot the factor loadings
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.plot(factor_loadings[:, i], label=f'Principal Component {i+1}', linewidth=2)

plt.xticks(ticks=np.arange(len(df_Yield.columns)), labels=df_Yield.columns)
plt.xlabel("Term")
plt.xticks(rotation = 45, ha='right')
plt.ylabel("Factor Loadings")
plt.legend()
plt.title("PCA of Yield Curve Change")
plt.show()

# Summary of PCA variance
print("Explained variance by component:", pca.explained_variance_ratio_)
print("Cumulative explained variance:", np.cumsum(pca.explained_variance_ratio_))



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_Yield_diff_standardized = scaler.fit_transform(df_Yield_diff)



pca = PCA()
pca.fit(df_Yield_diff_standardized)

# Summary DataFrame
summary_df = pd.DataFrame({
    "Standard deviation": np.sqrt(pca.explained_variance_),
    "Proportion of Variance": pca.explained_variance_ratio_,
    "Cumulative Proportion": np.cumsum(pca.explained_variance_ratio_)
})

# Display the summary table
print(summary_df)

#OUUUU????


# Summary DataFrame
summary_df = pd.DataFrame({
    "Standard deviation": np.sqrt(pca.explained_variance_),
    "Proportion of Variance": pca.explained_variance_ratio_,
    "Cumulative Proportion": np.cumsum(pca.explained_variance_ratio_)
})

# Display the summary table, similar to the R output
print(summary_df)



#%%%%

from statsmodels.tsa.statespace.mlemodel import MLEModel
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
import statsmodels.api as sm



#Define the Extended Nelson-Siegel Model Class
#To set up this model, create a custom MLEModel class in Python:


class NelsonSiegelExtended(MLEModel):
    def __init__(self, endog, macro_factors, lambda_value=0.0609):
        k_states = 3  # Number of latent factors (L_t, S_t, C_t)
        k_endog = len(endog[0])  # Number of observed maturities (tau)

        # Initialize the superclass with k_states and k_endog
        super(NelsonSiegelExtended, self).__init__(endog, k_states=k_states, k_posdef=k_states)

        # Store additional parameters
        self.macro_factors = macro_factors
        self.lambda_value = lambda_value

        # Explicitly initialize state-space matrices
        self.initialize_statespace()

        # Define tau values (maturities)
        tau = np.array([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])

        # Define the design matrix Z
        Z = np.zeros((len(tau), k_states))
        Z[:, 0] = 1  # Level factor
        Z[:, 1] = (1 - np.exp(-self.lambda_value * tau)) / (self.lambda_value * tau)  # Slope factor
        Z[:, 2] = ((1 - np.exp(-self.lambda_value * tau)) / (self.lambda_value * tau)) - np.exp(-self.lambda_value * tau)  # Curvature factor
        self['design'] = Z

        # Transition matrix for AR(1) dynamics (identity matrix)
        self['transition'] = np.eye(k_states)

        # Initialize the state covariance matrix
        self['state_cov'] = np.eye(k_states) * 0.1

        # Observation covariance matrix (measurement noise)
        self['obs_cov'] = np.eye(k_endog) * 0.1

        # Set initial state mean and covariance explicitly
        initial_state_mean = np.zeros(k_states)
        initial_state_cov = np.eye(k_states) * 0.1
        self.ssm.initialize_known(initial_state_mean, initial_state_cov)

    @property
    def param_names(self):
        return ['ar1_L', 'ar1_S', 'ar1_C', 'var_L', 'var_S', 'var_C', 'obs_var']


        
    def update(self, params, transformed=True, **kwargs):
        # Set up state transition matrix for AR(1)
        A = np.zeros((3, 3))
        A[np.diag_indices(3)] = params[:3]
        self['transition'] = A
        
        # Set measurement matrix for yield curve factors
        tau = np.array([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])  # Example maturities
        Z = np.zeros((len(tau), 3))
        Z[:, 0] = 1  # Level factor
        Z[:, 1] = (1 - np.exp(-self.lambda_value * tau)) / (self.lambda_value * tau)  # Slope factor
        Z[:, 2] = ((1 - np.exp(-self.lambda_value * tau)) / (self.lambda_value * tau)) - np.exp(-self.lambda_value * tau)  # Curvature factor
        self['design'] = Z
        
        # Set state covariance matrix
        Q = np.eye(3) * params[3:6]
        self['state_cov'] = Q
        
        # Set observation covariance matrix
        R = np.eye(len(tau)) * params[6]
        self['obs_cov'] = R



#Prepare the Macroeconomic Factor Matrix
#Extract the macroeconomic variables from your merged data for use in the model.


# Extract macroeconomic factors
macro_factors = merged_data[['GDP', 'CPI_QoQ_LogChange', 'Unemployment']].values
yields = merged_data[["Taux de l'Echéance Constante - 1 an", "Taux de l'Echéance Constante - 2 ans", 
 "Taux de l'Echéance Constante - 3 ans", "Taux de l'Echéance Constante - 5 ans", 
 "Taux de l'Echéance Constante - 7 ans", "Taux de l'Echéance Constante - 10 ans", 
 "Taux de l'Echéance Constante - 15 ans", "Taux de l'Echéance Constante - 20 ans", 
 "Taux de l'Echéance Constante - 25 ans", "Taux de l'Echéance Constante - 30 ans"]].values



#Estimate the Model Parameters
#Fit the state-space model to the yield data and macroeconomic variables.

# Initialize model
lambda_value = 0.0609  # Suggested value for Nelson-Siegel model
model = NelsonSiegelExtended(yields, macro_factors, lambda_value=lambda_value)

# Define initial parameters (these may require tuning)
initial_params = np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1])

# Fit model
results = model.fit(initial_params, disp=False)
print(results.summary())



#Plot the Estimated Factors
#After fitting, plot the estimated latent factors to interpret how they align with the macroeconomic conditions

import matplotlib.pyplot as plt

# Extract estimated factors
level_factor = results.filtered_state[0]
slope_factor = results.filtered_state[1]
curvature_factor = results.filtered_state[2]

# Plot factors
plt.figure(figsize=(10, 6))
plt.plot(level_factor, label='Level')
plt.plot(slope_factor, label='Slope')
plt.plot(curvature_factor, label='Curvature')
plt.legend()
plt.title('Estimated Yield Curve Factors')
plt.xlabel('Time')
plt.ylabel('Factor Value')
plt.show()






#Extract the Factor Series
#After fitting the model, you can extract the estimated time series for the Level, Slope, and Curvature factors from the filtered_state attributes of the results.

# Extract the estimated factors
level_factor = results.filtered_state[0]
slope_factor = results.filtered_state[1]
curvature_factor = results.filtered_state[2]


#Plot the Factor Time Series
#Visualize each factor over time to understand how they behave during different economic periods. This helps in identifying trends, structural changes, or cyclical behaviors.


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

# Plot Level
plt.subplot(3, 1, 1)
plt.plot(level_factor, label='Level Factor (Lt)')
plt.title('Level Factor')
plt.xlabel('Time')
plt.ylabel('Level')
plt.legend()

# Plot Slope
plt.subplot(3, 1, 2)
plt.plot(slope_factor, label='Slope Factor (St)')
plt.title('Slope Factor')
plt.xlabel('Time')
plt.ylabel('Slope')
plt.legend()

# Plot Curvature
plt.subplot(3, 1, 3)
plt.plot(curvature_factor, label='Curvature Factor (Ct)')
plt.title('Curvature Factor')
plt.xlabel('Time')
plt.ylabel('Curvature')
plt.legend()

plt.tight_layout()
plt.show()


#Correlation Analysis with Macroeconomic Variables
#To understand how these factors relate to macroeconomic conditions, calculate and interpret the correlations between each factor and the macroeconomic variables (GDP growth, CPI, and Unemployment).

import pandas as pd

# Combine factors and macroeconomic data into a single DataFrame for correlation analysis
factor_data = pd.DataFrame({
    'Level': level_factor,
    'Slope': slope_factor,
    'Curvature': curvature_factor,
    'GDP_Growth': merged_data['GDP'].values,
    'CPI_Inflation': merged_data['CPI_QoQ_LogChange'].values,
    'Unemployment': merged_data['Unemployment'].values
})

# Calculate correlations
correlations = factor_data.corr()
pd.set_option("display.max_columns", None)
print(correlations)



#%%%%%
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.kalman_filter import KalmanFilter
from statsmodels.tsa.statespace.mlemodel import MLEModel
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Assuming your data is in a dataframe `data`
# Extract necessary columns
yields = merged_data[
    ["Taux de l'Echéance Constante - 1 an", "Taux de l'Echéance Constante - 2 ans", 
     "Taux de l'Echéance Constante - 3 ans", "Taux de l'Echéance Constante - 5 ans", 
     "Taux de l'Echéance Constante - 7 ans", "Taux de l'Echéance Constante - 10 ans", 
     "Taux de l'Echéance Constante - 15 ans", "Taux de l'Echéance Constante - 20 ans", 
     "Taux de l'Echéance Constante - 25 ans", "Taux de l'Echéance Constante - 30 ans"]
]
gdp = merged_data["GDP"]
cpi = merged_data["CPI"]
unemployment = merged_data["Unemployment"]
lambda_param = merged_data["lambda"].iloc[0]  # Assume lambda is constant, or choose first value as starting point

# Correct the number of parameters for initial_params to match model requirements
initial_params = np.ones(25)  # 9 for transition matrix + 6 for state_cov + 10 for obs_cov

# Update the model class for Extended Nelson-Siegel
class ExtendedNelsonSiegel(MLEModel):
    def __init__(self, yields, gdp, cpi, unemployment, lambda_param):
        k_states = 6  # Level, Slope, Curvature, GDP, CPI, Unemployment
        k_posdef = 6  # Allow covariance for these six factors

        super(ExtendedNelsonSiegel, self).__init__(endog=yields, k_states=k_states, k_posdef=k_posdef)
        
        # Set parameters
        self.lambda_param = lambda_param
        
        # Explanatory macro factors
        self.macro_factors = np.column_stack([gdp, cpi, unemployment])
        
        # Initialize matrices
        self['design'] = self.design_matrix(yields.columns)  # Measurement equation design matrix
        self['transition'] = np.eye(k_states)  # Transition matrix for AR(1) dynamics
        self['selection'] = np.eye(k_states)  # Identity matrix for process noise selection
        
        # Initialize state vector and state covariance
        self.initialize_state()

    def design_matrix(self, maturities):
        """ Creates the design matrix Λ for the measurement equation based on NS formula. """
        maturities = np.array([1, 2, 3, 5, 7, 10, 15, 20, 25, 30])  # Years for constant maturities
        ns_loadings = np.column_stack([
            np.ones(len(maturities)),  # Level
            (1 - np.exp(-self.lambda_param * maturities)) / (self.lambda_param * maturities),  # Slope
            ((1 - np.exp(-self.lambda_param * maturities)) / (self.lambda_param * maturities) - np.exp(-self.lambda_param * maturities))  # Curvature
        ])
        return np.hstack([ns_loadings, np.zeros((len(maturities), 3))])  # Add columns for macro factors

    def update(self, params, **kwargs):
        # Update transition matrix with AR(1) dynamics for yield curve factors
        self['transition', :3, :3] = params[:9].reshape(3, 3)
        
        # Set covariance matrices for process noise and measurement errors
        self['state_cov'] = np.diag(params[9:15])  # Variance of state innovations
        self['obs_cov'] = np.diag(params[15:25])  # 10 diagonal elements for observation covariance matrix

    def initialize_state(self):
        # Explicit initialization of the initial state mean and covariance
        self.ssm.initialize_approximate_diffuse()  # Initializes states with diffuse priors
        self.ssm.initial_state_cov = np.eye(self.k_states)  # Identity for initial state covariance

    def loglike(self, params, **kwargs):
        # Update parameters and compute log likelihood
        self.update(params)
        return super(ExtendedNelsonSiegel, self).loglike(params, **kwargs)




# Initialize and fit the model
model = ExtendedNelsonSiegel(yields=yields, gdp=gdp, cpi=cpi, unemployment=unemployment, lambda_param=lambda_param)
result = minimize(model.loglike, initial_params, method='L-BFGS-B')
model.update(result.x)



# State estimates (filtered estimates for level, slope, curvature, and macro factors)
filtered_states = model.filter(result.x).filtered_state

# Plot estimated factors
plt.figure(figsize=(10, 6))
plt.plot(filtered_states[0], label="Level (Lt)")
plt.plot(filtered_states[1], label="Slope (St)")
plt.plot(filtered_states[2], label="Curvature (Ct)")
plt.legend()
plt.title("Estimated Yield Curve Factors (Level, Slope, Curvature)")
plt.show()

# Plot macro factors
plt.figure(figsize=(10, 6))
plt.plot(filtered_states[3], label="GDP")
plt.plot(filtered_states[4], label="CPI")
plt.plot(filtered_states[5], label="Unemployment")
plt.legend()
plt.title("Macroeconomic Factors")
plt.show()
