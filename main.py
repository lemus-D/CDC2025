import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, HuberRegressor
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy import stats
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("Set2")

file_name = "Business.xlsx"
sheet_to_load = "Table 1"

# --- 1. Load and clean the data (same as before) ---
df = pd.read_excel(file_name, sheet_name=sheet_to_load, header=None, skiprows=5)
new_headers = df.iloc[0] 
df = df[1:] 
df.columns = new_headers
columns_to_convert = df.columns[2:]
for column in columns_to_convert:
    df[column] = pd.to_numeric(df[column], errors='coerce')
df.dropna(how='all', inplace=True)

# --- 2. Extract Private sector and calculate Government (same as before) ---
industry_series = df.iloc[:, 1].astype(str)
private_row_idx = None
total_space_economy_row_idx = None

for idx, industry in enumerate(industry_series):
    industry_clean = industry.strip().lower()
    if 'private' in industry_clean and ('industries' in industry_clean or 'sector' in industry_clean or industry_clean == 'private'):
        private_row_idx = idx
        break

if private_row_idx is not None:
    private_totals = pd.to_numeric(df.iloc[private_row_idx, 2:], errors='coerce')
    private_totals.index = columns_to_convert
    
    # Calculate total and government
    total_space_economy = df.iloc[0, 2:] if df.shape[0] > 0 else df.iloc[:, 2:].sum()
    total_space_economy = pd.to_numeric(total_space_economy, errors='coerce')
    total_space_economy.index = columns_to_convert
    
    government_totals = total_space_economy - private_totals

# --- 3. IMPROVED FORECASTING FUNCTIONS ---

class ImprovedSpaceEconomyForecaster:
    def __init__(self, data, sector_name, forecast_years=5):
        self.data = data.dropna()
        self.sector_name = sector_name
        self.forecast_years = forecast_years
        
        # Prepare data
        self.years = np.array([int(year) for year in self.data.index])
        self.values = self.data.values
        self.future_years = np.arange(self.years[-1] + 1, self.years[-1] + forecast_years + 1)
        
        # Results storage
        self.models = {}
        self.predictions = {}
        self.model_scores = {}
    
    def detect_outliers_and_trends(self):
        """Analyze data characteristics to choose appropriate models"""
        # Calculate growth rates
        growth_rates = np.diff(self.values) / self.values[:-1] * 100
        
        # Detect outliers using IQR method
        Q1, Q3 = np.percentile(growth_rates, [25, 75])
        IQR = Q3 - Q1
        outlier_threshold = 1.5 * IQR
        outliers = np.abs(growth_rates - np.median(growth_rates)) > outlier_threshold
        
        # Trend analysis
        slope, _, r_value, _, _ = stats.linregress(self.years, self.values)
        
        print(f"\n--- {self.sector_name} Data Analysis ---")
        print(f"Average annual growth: {np.mean(growth_rates):.2f}%")
        print(f"Growth volatility (std): {np.std(growth_rates):.2f}%")
        print(f"Linear trend R²: {r_value**2:.3f}")
        print(f"Outlier years detected: {np.sum(outliers)}")
        
        return {
            'avg_growth': np.mean(growth_rates),
            'volatility': np.std(growth_rates),
            'trend_r2': r_value**2,
            'outliers': outliers,
            'slope': slope
        }
    
    def fit_robust_linear(self):
        """Robust linear regression that handles outliers better"""
        X = self.years.reshape(-1, 1)
        
        # Huber regressor - robust to outliers
        huber = HuberRegressor(epsilon=1.35, alpha=0.0)
        huber.fit(X, self.values)
        
        # Predictions
        future_X = self.future_years.reshape(-1, 1)
        predictions = huber.predict(future_X)
        
        # Score
        y_pred = huber.predict(X)
        score = r2_score(self.values, y_pred)
        mae = mean_absolute_error(self.values, y_pred)
        
        self.models['Robust_Linear'] = huber
        self.predictions['Robust_Linear'] = predictions
        self.model_scores['Robust_Linear'] = {'R²': score, 'MAE': mae}
        
        return predictions
    
    def fit_piecewise_regression(self):
        """Piecewise regression to handle structural breaks"""
        def piecewise_linear(x, x0, y0, k1, k2):
            return np.piecewise(x, [x <= x0, x > x0], 
                              [lambda x: k1*x + y0-k1*x0, lambda x: k2*x + y0-k2*x0])
        
        try:
            # Find the best breakpoint
            mid_year = self.years[len(self.years)//2]
            p0 = [mid_year, np.mean(self.values), 0, 0]  # initial guess
            
            popt, _ = curve_fit(piecewise_linear, self.years, self.values, p0=p0, maxfev=5000)
            
            # Predict future
            predictions = piecewise_linear(self.future_years, *popt)
            
            # Score
            y_pred = piecewise_linear(self.years, *popt)
            score = r2_score(self.values, y_pred)
            mae = mean_absolute_error(self.values, y_pred)
            
            self.models['Piecewise'] = popt
            self.predictions['Piecewise'] = predictions
            self.model_scores['Piecewise'] = {'R²': score, 'MAE': mae, 'Breakpoint': popt[0]}
            
            return predictions
        except:
            print(f"Piecewise regression failed for {self.sector_name}")
            return None
    
    def fit_exponential_with_cycles(self):
        """Exponential growth with cyclical component"""
        def exp_cyclical(x, a, b, c, d, period):
            return a * np.exp(b * (x - x[0])) + c * np.sin(2 * np.pi * (x - x[0]) / period) + d
        
        try:
            # Initial parameters
            p0 = [self.values[0], 0.02, 1000, self.values[0], 8]  # 8-year cycle
            
            popt, _ = curve_fit(exp_cyclical, self.years, self.values, p0=p0, maxfev=10000)
            
            predictions = exp_cyclical(self.future_years, *popt)
            
            # Score
            y_pred = exp_cyclical(self.years, *popt)
            score = r2_score(self.values, y_pred)
            mae = mean_absolute_error(self.values, y_pred)
            
            self.models['Exp_Cyclical'] = popt
            self.predictions['Exp_Cyclical'] = predictions
            self.model_scores['Exp_Cyclical'] = {'R²': score, 'MAE': mae}
            
            return predictions
        except:
            print(f"Exponential cyclical failed for {self.sector_name}")
            return None
    
    def fit_random_forest(self):
        """Random Forest with engineered features"""
        # Create features
        X = np.column_stack([
            self.years - self.years[0],  # Years since start
            np.arange(len(self.years)),  # Time index
            (self.years - self.years[0]) ** 2,  # Quadratic time
            np.sin(2 * np.pi * (self.years - self.years[0]) / 8),  # 8-year cycle
            np.cos(2 * np.pi * (self.years - self.years[0]) / 8),  # 8-year cycle
        ])
        
        # Fit model
        rf = RandomForestRegressor(n_estimators=200, random_state=42, max_depth=5)
        rf.fit(X, self.values)
        
        # Future features
        future_X = np.column_stack([
            self.future_years - self.years[0],
            np.arange(len(self.years), len(self.years) + len(self.future_years)),
            (self.future_years - self.years[0]) ** 2,
            np.sin(2 * np.pi * (self.future_years - self.years[0]) / 8),
            np.cos(2 * np.pi * (self.future_years - self.years[0]) / 8),
        ])
        
        predictions = rf.predict(future_X)
        
        # Score
        y_pred = rf.predict(X)
        score = r2_score(self.values, y_pred)
        mae = mean_absolute_error(self.values, y_pred)
        
        self.models['Random_Forest'] = rf
        self.predictions['Random_Forest'] = predictions
        self.model_scores['Random_Forest'] = {'R²': score, 'MAE': mae}
        
        return predictions
    
    def fit_trend_plus_noise(self):
        """Separate trend from cyclical/noise components"""
        from scipy.signal import detrend
        
        # Detrend the data
        detrended = detrend(self.values)
        
        # Fit trend
        trend_model = LinearRegression()
        X = self.years.reshape(-1, 1)
        trend_model.fit(X, self.values - detrended)
        
        # Predict trend
        future_X = self.future_years.reshape(-1, 1)
        trend_prediction = trend_model.predict(future_X)
        
        # Assume cyclical component dampens over time
        cycle_damping = 0.8  # Cycles reduce by 20% per year
        last_cycles = detrended[-3:].mean()  # Average recent cyclical component
        
        future_cycles = [last_cycles * (cycle_damping ** i) for i in range(1, len(self.future_years) + 1)]
        
        predictions = trend_prediction + future_cycles
        
        # Score on historical data
        historical_pred = trend_model.predict(X)
        score = r2_score(self.values, historical_pred)
        mae = mean_absolute_error(self.values, historical_pred)
        
        self.predictions['Trend_Plus_Cycles'] = predictions
        self.model_scores['Trend_Plus_Cycles'] = {'R²': score, 'MAE': mae}
        
        return predictions
    
    def create_ensemble(self):
        """Create weighted ensemble based on model performance"""
        if len(self.predictions) < 2:
            return None
        
        # Get R² scores
        weights = []
        models = []
        for name, score_dict in self.model_scores.items():
            if name in self.predictions and 'R²' in score_dict:
                r2 = max(0, score_dict['R²'])  # Ensure non-negative
                weights.append(r2)
                models.append(name)
        
        if not weights or sum(weights) == 0:
            # Simple average if no good scores
            predictions_list = [self.predictions[name] for name in self.predictions.keys()]
            ensemble = np.mean(predictions_list, axis=0)
        else:
            # Weighted average
            weights = np.array(weights) / sum(weights)
            predictions_list = [self.predictions[name] for name in models]
            ensemble = np.average(predictions_list, weights=weights, axis=0)
        
        self.predictions['Ensemble'] = ensemble
        return ensemble
    
    def fit_all_models(self):
        """Fit all models and create ensemble"""
        print(f"\n=== Fitting Models for {self.sector_name} ===")
        
        # Analyze data
        analysis = self.detect_outliers_and_trends()
        
        # Fit models
        self.fit_robust_linear()
        self.fit_piecewise_regression()
        self.fit_exponential_with_cycles()
        self.fit_random_forest()
        self.fit_trend_plus_noise()
        
        # Create ensemble
        self.create_ensemble()
        
        # Print results
        print(f"\nModel Performance for {self.sector_name}:")
        print("-" * 50)
        for name, scores in self.model_scores.items():
            r2 = scores.get('R²', 0)
            mae = scores.get('MAE', 0)
            print(f"{name:20} R²: {r2:.3f}, MAE: ${mae:,.0f}M")
    
    def plot_forecasts(self):
        """Plot all forecasts"""
        plt.figure(figsize=(12, 8))
        
        # Historical data
        plt.plot(self.years, self.values, 'o-', linewidth=3, markersize=8, 
                label=f'{self.sector_name} Historical', color='black')
        
        # Model predictions
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']
        
        for i, (name, pred) in enumerate(self.predictions.items()):
            color = colors[i % len(colors)]
            linestyle = '--' if name != 'Ensemble' else '-'
            linewidth = 3 if name == 'Ensemble' else 2
            alpha = 0.9 if name == 'Ensemble' else 0.7
            
            plt.plot(self.future_years, pred, linestyle, linewidth=linewidth, 
                    label=f'{name}', color=color, alpha=alpha)
        
        plt.axvline(x=self.years[-1], color='gray', linestyle=':', alpha=0.5, 
                   label='Forecast Start')
        
        plt.xlabel('Year')
        plt.ylabel('Value (Millions $)')
        plt.title(f'{self.sector_name} - Improved Forecasting Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# --- 4. Run Improved Analysis ---
if 'private_totals' in locals() and 'government_totals' in locals():
    
    # Analyze Private Sector (original 5-year forecast)
    private_forecaster = ImprovedSpaceEconomyForecaster(private_totals, "Private Sector")
    private_forecaster.fit_all_models()
    private_forecaster.plot_forecasts()
    
    # Analyze Government Sector (original 5-year forecast)
    government_forecaster = ImprovedSpaceEconomyForecaster(government_totals, "Government Sector")
    government_forecaster.fit_all_models()
    government_forecaster.plot_forecasts()
    
    # Combined comparison plot (original)
    plt.figure(figsize=(15, 8))
    
    # Historical data
    plt.plot(private_totals.index.astype(int), private_totals.values, 'o-', 
             linewidth=3, markersize=8, label='Private Historical', color='red')
    plt.plot(government_totals.index.astype(int), government_totals.values, 'o-', 
             linewidth=3, markersize=8, label='Government Historical', color='blue')
    
    # Best forecasts (Ensemble)
    future_years = private_forecaster.future_years
    plt.plot(future_years, private_forecaster.predictions['Ensemble'], '--', 
             linewidth=3, label='Private Forecast (Ensemble)', color='red', alpha=0.8)
    plt.plot(future_years, government_forecaster.predictions['Ensemble'], '--', 
             linewidth=3, label='Government Forecast (Ensemble)', color='blue', alpha=0.8)
    
    plt.axvline(x=2023, color='gray', linestyle=':', alpha=0.5, label='Forecast Start')
    plt.xlabel('Year')
    plt.ylabel('Value (Millions $)')
    plt.title('Space Economy Forecasting - 5-Year Outlook (2028)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # --- NEW: Extended 10-year forecasts to 2033 ---
    print("\n" + "="*80)
    print("GENERATING 10-YEAR FORECASTS TO 2033")
    print("="*80)
    
    private_forecaster_2033 = ImprovedSpaceEconomyForecaster(private_totals, "Private Sector", forecast_years=10)
    private_forecaster_2033.fit_all_models()
    
    government_forecaster_2033 = ImprovedSpaceEconomyForecaster(government_totals, "Government Sector", forecast_years=10)
    government_forecaster_2033.fit_all_models()
    
    # 10-year forecast plots
    private_forecaster_2033.plot_forecasts()
    government_forecaster_2033.plot_forecasts()
    
    # Combined 10-year plot
    plt.figure(figsize=(16, 9))
    
    # Historical data
    plt.plot(private_totals.index.astype(int), private_totals.values, 'o-', 
             linewidth=3, markersize=8, label='Private Historical', color='red')
    plt.plot(government_totals.index.astype(int), government_totals.values, 'o-', 
             linewidth=3, markersize=8, label='Government Historical', color='blue')
    
    # 10-year forecasts
    future_years_2033 = private_forecaster_2033.future_years
    plt.plot(future_years_2033, private_forecaster_2033.predictions['Ensemble'], '--', 
             linewidth=3, label='Private Forecast to 2033', color='red', alpha=0.8)
    plt.plot(future_years_2033, government_forecaster_2033.predictions['Ensemble'], '--', 
             linewidth=3, label='Government Forecast to 2033', color='blue', alpha=0.8)
    
    plt.axvline(x=2023, color='gray', linestyle=':', alpha=0.5, label='Forecast Start')
    plt.xlabel('Year')
    plt.ylabel('Value (Millions $)')
    plt.title('Space Economy Forecasting - 10-Year Outlook (2033)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # --- NEW: Extended 30-year forecasts to 2053 ---
    print("\n" + "="*80)
    print("GENERATING 30-YEAR FORECASTS TO 2053")
    print("="*80)
    
    private_forecaster_2053 = ImprovedSpaceEconomyForecaster(private_totals, "Private Sector", forecast_years=30)
    private_forecaster_2053.fit_all_models()
    
    government_forecaster_2053 = ImprovedSpaceEconomyForecaster(government_totals, "Government Sector", forecast_years=30)
    government_forecaster_2053.fit_all_models()
    
    # 30-year forecast plots
    private_forecaster_2053.plot_forecasts()
    government_forecaster_2053.plot_forecasts()
    
    # Combined 30-year plot
    plt.figure(figsize=(18, 10))
    
    # Historical data
    plt.plot(private_totals.index.astype(int), private_totals.values, 'o-', 
             linewidth=3, markersize=8, label='Private Historical', color='red')
    plt.plot(government_totals.index.astype(int), government_totals.values, 'o-', 
             linewidth=3, markersize=8, label='Government Historical', color='blue')
    
    # 30-year forecasts
    future_years_2053 = private_forecaster_2053.future_years
    plt.plot(future_years_2053, private_forecaster_2053.predictions['Ensemble'], '--', 
             linewidth=3, label='Private Forecast to 2053', color='red', alpha=0.8)
    plt.plot(future_years_2053, government_forecaster_2053.predictions['Ensemble'], '--', 
             linewidth=3, label='Government Forecast to 2053', color='blue', alpha=0.8)
    
    plt.axvline(x=2023, color='gray', linestyle=':', alpha=0.5, label='Forecast Start')
    plt.xlabel('Year')
    plt.ylabel('Value (Millions $)')
    plt.title('Space Economy Forecasting - 30-Year Outlook (2053)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # --- NEW: Comprehensive comparison across all time horizons ---
    plt.figure(figsize=(20, 12))
    
    # Create subplots for different time horizons
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Historical + 5-year
    ax1.plot(private_totals.index.astype(int), private_totals.values, 'o-', 
             linewidth=2, markersize=6, label='Private Historical', color='red')
    ax1.plot(government_totals.index.astype(int), government_totals.values, 'o-', 
             linewidth=2, markersize=6, label='Government Historical', color='blue')
    ax1.plot(private_forecaster.future_years, private_forecaster.predictions['Ensemble'], 
             '--', linewidth=2, label='Private 2028', color='red', alpha=0.8)
    ax1.plot(government_forecaster.future_years, government_forecaster.predictions['Ensemble'], 
             '--', linewidth=2, label='Government 2028', color='blue', alpha=0.8)
    ax1.axvline(x=2023, color='gray', linestyle=':', alpha=0.5)
    ax1.set_title('5-Year Forecast (2028)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Value (Millions $)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Historical + 10-year
    ax2.plot(private_totals.index.astype(int), private_totals.values, 'o-', 
             linewidth=2, markersize=6, label='Private Historical', color='red')
    ax2.plot(government_totals.index.astype(int), government_totals.values, 'o-', 
             linewidth=2, markersize=6, label='Government Historical', color='blue')
    ax2.plot(future_years_2033, private_forecaster_2033.predictions['Ensemble'], 
             '--', linewidth=2, label='Private 2033', color='red', alpha=0.8)
    ax2.plot(future_years_2033, government_forecaster_2033.predictions['Ensemble'], 
             '--', linewidth=2, label='Government 2033', color='blue', alpha=0.8)
    ax2.axvline(x=2023, color='gray', linestyle=':', alpha=0.5)
    ax2.set_title('10-Year Forecast (2033)')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Value (Millions $)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Historical + 30-year
    ax3.plot(private_totals.index.astype(int), private_totals.values, 'o-', 
             linewidth=2, markersize=6, label='Private Historical', color='red')
    ax3.plot(government_totals.index.astype(int), government_totals.values, 'o-', 
             linewidth=2, markersize=6, label='Government Historical', color='blue')
    ax3.plot(future_years_2053, private_forecaster_2053.predictions['Ensemble'], 
             '--', linewidth=2, label='Private 2053', color='red', alpha=0.8)
    ax3.plot(future_years_2053, government_forecaster_2053.predictions['Ensemble'], 
             '--', linewidth=2, label='Government 2053', color='blue', alpha=0.8)
    ax3.axvline(x=2023, color='gray', linestyle=':', alpha=0.5)
    ax3.set_title('30-Year Forecast (2053)')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Value (Millions $)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Growth trajectory comparison
    all_years = list(private_totals.index.astype(int)) + list(future_years_2053)
    private_all = list(private_totals.values) + list(private_forecaster_2053.predictions['Ensemble'])
    gov_all = list(government_totals.values) + list(government_forecaster_2053.predictions['Ensemble'])
    total_all = [p + g for p, g in zip(private_all, gov_all)]
    
    ax4.plot(all_years, total_all, 'o-', linewidth=3, markersize=4, 
             label='Total Space Economy', color='purple')
    ax4.plot(all_years, private_all, '--', linewidth=2, 
             label='Private Sector', color='red', alpha=0.7)
    ax4.plot(all_years, gov_all, '--', linewidth=2, 
             label='Government Sector', color='blue', alpha=0.7)
    ax4.axvline(x=2023, color='gray', linestyle=':', alpha=0.5)
    ax4.set_title('Complete Growth Trajectory (2012-2053)')
    ax4.set_xlabel('Year')
    ax4.set_ylabel('Value (Millions $)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Space Economy Multi-Horizon Forecasting Analysis', fontsize=16, y=0.98)
    plt.tight_layout()
    plt.show()
    
    # Print comprehensive summary
    print("\n" + "="*80)
    print("COMPREHENSIVE FORECAST SUMMARY")
    print("="*80)
    
    # 2028 forecasts
    print("\n2028 FORECASTS (5-year):")
    print(f"Private Sector: ${private_forecaster.predictions['Ensemble'][-1]:,.0f}M")
    print(f"Government Sector: ${government_forecaster.predictions['Ensemble'][-1]:,.0f}M")
    total_2028 = private_forecaster.predictions['Ensemble'][-1] + government_forecaster.predictions['Ensemble'][-1]
    print(f"Total Space Economy: ${total_2028:,.0f}M")
    
    # 2033 forecasts
    print("\n2033 FORECASTS (10-year):")
    print(f"Private Sector: ${private_forecaster_2033.predictions['Ensemble'][-1]:,.0f}M")
    print(f"Government Sector: ${government_forecaster_2033.predictions['Ensemble'][-1]:,.0f}M")
    total_2033 = private_forecaster_2033.predictions['Ensemble'][-1] + government_forecaster_2033.predictions['Ensemble'][-1]
    print(f"Total Space Economy: ${total_2033:,.0f}M")
    
    # 2053 forecasts
    print("\n2053 FORECASTS (30-year):")
    print(f"Private Sector: ${private_forecaster_2053.predictions['Ensemble'][-1]:,.0f}M")
    print(f"Government Sector: ${government_forecaster_2053.predictions['Ensemble'][-1]:,.0f}M")
    total_2053 = private_forecaster_2053.predictions['Ensemble'][-1] + government_forecaster_2053.predictions['Ensemble'][-1]
    print(f"Total Space Economy: ${total_2053:,.0f}M")
    
    # Growth rates
    current_total = private_totals.iloc[-1] + government_totals.iloc[-1]
    cagr_2028 = ((total_2028 / current_total) ** (1/5) - 1) * 100
    cagr_2033 = ((total_2033 / current_total) ** (1/10) - 1) * 100
    cagr_2053 = ((total_2053 / current_total) ** (1/30) - 1) * 100
    
    print(f"\nPROJECTED COMPOUND ANNUAL GROWTH RATES:")
    print(f"2023-2028 (5-year):  {cagr_2028:.1f}%")
    print(f"2023-2033 (10-year): {cagr_2033:.1f}%")
    print(f"2023-2053 (30-year): {cagr_2053:.1f}%")
    
    # Market share evolution
    print(f"\nMARKET SHARE EVOLUTION:")
    current_private_share = private_totals.iloc[-1] / current_total * 100
    share_2028 = private_forecaster.predictions['Ensemble'][-1] / total_2028 * 100
    share_2033 = private_forecaster_2033.predictions['Ensemble'][-1] / total_2033 * 100
    share_2053 = private_forecaster_2053.predictions['Ensemble'][-1] / total_2053 * 100
    
    print(f"Current (2023): Private {current_private_share:.1f}%, Government {100-current_private_share:.1f}%")
    print(f"2028: Private {share_2028:.1f}%, Government {100-share_2028:.1f}%")
    print(f"2033: Private {share_2033:.1f}%, Government {100-share_2033:.1f}%")
    print(f"2053: Private {share_2053:.1f}%, Government {100-share_2053:.1f}%")
    
else:
    print("Could not load data properly. Please check your Excel file structure.")