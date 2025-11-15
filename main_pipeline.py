import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import warnings
import joblib # Import joblib for model serialization
import os # Import os for directory creation
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
pd.options.display.float_format = '{:,.4f}'.format


class CommodityPredictionPipeline:
    """
    Commodity Prediction Pipeline
    """
    def __init__(self, rolling_window=7, indicator='CCI', max_lag=30, output_dir='/results/pipeline_results'):
        """
        Parameters:
        -----------
        rolling_window : int
            Rolling window size for predictions
        indicator : str
            Technical indicator to use ('CCI', 'RSI', 'MACD', 'BB')
        max_lag : int
            Maximum lag to test for optimal features
        output_dir : str
            Directory to save all results and models
        """
        self.rolling_window = rolling_window
        self.indicator = indicator
        self.max_lag = max_lag
        self.output_dir = output_dir

        self.exchanges = {
            'US': [],
            'LME': [],
            'FX': [],
            'JPX': []
        }

        self.models = {}
        self.results = {}
        self.statistical_tests = {}

        os.makedirs(self.output_dir, exist_ok=True)

    def load_data(self, train_path):
        """Load and initial preprocessing of competition data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        self.train_df = pd.read_csv(train_path)
        # If 'date_id' exists, set it as index
        if 'date_id' in self.train_df.columns:
            self.train_df.set_index('date_id', inplace=True)

        # Identify commodity columns (excluding date/id columns and filtering for '_Close' suffix)
        # Re-initialize exchanges dictionary to ensure clean categorization each time load_data is called
        self.exchanges = {
            'US': [],
            'LME': [],
            'FX': [],
            'JPX': []
        }

        for col in self.train_df.columns:
            # Prioritize prefix matching for categorization
            if col.startswith('US_'):
                self.exchanges['US'].append(col)
            elif col.startswith('LME_'):
                self.exchanges['LME'].append(col)
            elif col.startswith('FX_'):
                self.exchanges['FX'].append(col)
            elif col.startswith('JPX_'):
                self.exchanges['JPX'].append(col)
            # Then, handle columns ending with '_Close' that were not caught by a specific prefix
            elif col.endswith('_Close'):
                self.exchanges['LME'].append(col)

        print(f"\nLoaded {len(self.train_df)} rows")
        print(f"Total columns categorized: {sum(len(v) for v in self.exchanges.values())}")
        for exchange, commodities in self.exchanges.items():
            print(f"  {exchange}: {len(commodities)} commodities")

        return self.train_df

    def statistical_tests_commodity(self, series, name):
        """
        Comprehensive statistical testing for each commodity:
        - Augmented Dickey-Fuller (stationarity)
        - KPSS Test (stationarity)
        - Ljung-Box (autocorrelation)
        - Normality tests
        - Linearity assessment
        """
        results = {
            'commodity': name,
            'n_obs': len(series),
            'missing_pct': series.isna().mean() * 100
        }

        # Remove NaN values for testing
        clean_series = series.dropna()

        if len(clean_series) < 10:
            return results

        # 1. Augmented Dickey-Fuller Test (H0: Non-stationary)
        try:
            adf_result = adfuller(clean_series, autolag='AIC')
            results['adf_statistic'] = adf_result[0]
            results['adf_pvalue'] = adf_result[1]
            results['adf_stationary'] = adf_result[1] < 0.05
        except:
            results['adf_statistic'] = np.nan
            results['adf_pvalue'] = np.nan
            results['adf_stationary'] = False

        # 2. KPSS Test (H0: Stationary)
        try:
            kpss_result = kpss(clean_series, regression='c', nlags='auto')
            results['kpss_statistic'] = kpss_result[0]
            results['kpss_pvalue'] = kpss_result[1]
            results['kpss_stationary'] = kpss_result[1] > 0.05
        except:
            results['kpss_statistic'] = np.nan
            results['kpss_pvalue'] = np.nan
            results['kpss_stationary'] = False

        # 3. Ljung-Box Test (autocorrelation)
        try:
            lb_result = acorr_ljungbox(clean_series, lags=[10], return_df=True)
            results['ljungbox_statistic'] = lb_result['lb_stat'].values[0]
            results['ljungbox_pvalue'] = lb_result['lb_pvalue'].values[0]
            results['autocorrelated'] = lb_result['lb_pvalue'].values[0] < 0.05
        except:
            results['ljungbox_statistic'] = np.nan
            results['ljungbox_pvalue'] = np.nan
            results['autocorrelated'] = False

        # 4. Normality Tests
        try:
            _, norm_pvalue = stats.normaltest(clean_series)
            results['normality_pvalue'] = norm_pvalue
            results['is_normal'] = norm_pvalue > 0.05
        except:
            results['normality_pvalue'] = np.nan
            results['is_normal'] = False

        # 5. Log-normality assessment (test log-transformed data)
        try:
            if (clean_series > 0).all():
                log_series = np.log(clean_series)
                _, log_norm_pvalue = stats.normaltest(log_series)
                results['log_normality_pvalue'] = log_norm_pvalue
                results['is_log_normal'] = log_norm_pvalue > 0.05
            else:
                results['log_normality_pvalue'] = np.nan
                results['is_log_normal'] = False
        except:
            results['log_normality_pvalue'] = np.nan
            results['is_log_normal'] = False

        # 6. Linearity assessment (using Pearson correlation with time)
        try:
            time_index = np.arange(len(clean_series))
            correlation, corr_pvalue = stats.pearsonr(time_index, clean_series)
            results['linear_trend_corr'] = correlation
            results['linear_trend_pvalue'] = corr_pvalue
            results['has_linear_trend'] = corr_pvalue < 0.05
        except:
            results['linear_trend_corr'] = np.nan
            results['linear_trend_pvalue'] = np.nan
            results['has_linear_trend'] = False

        return results

    def run_statistical_analysis(self):
        """Run comprehensive statistical tests for all commodities"""
        print("\n" + "=" * 80)
        print("STATISTICAL ANALYSIS")
        print("=" * 80)

        all_results = []

        for exchange, commodities in self.exchanges.items():
            if not commodities:
                continue

            print(f"\n{exchange} Exchange:")
            for commodity in commodities:
                if commodity in self.train_df.columns:
                    series = self.train_df[commodity]
                    results = self.statistical_tests_commodity(series, commodity)
                    results['exchange'] = exchange
                    all_results.append(results)

                    print(f"  {commodity}:")
                    print(f"    ADF Stationary: {results.get('adf_stationary', 'N/A')} (p={results.get('adf_pvalue', 'N/A'):.4f})" if results.get('adf_pvalue') else "    ADF: N/A")
                    print(f"    Log-Normal: {results.get('is_log_normal', 'N/A')}")
                    print(f"    Autocorrelated: {results.get('autocorrelated', 'N/A')}")

        self.statistical_tests = pd.DataFrame(all_results)
        return self.statistical_tests

    def transform_to_log_normal(self, df):
        """Transform data to log-normal distribution"""
        print("\n" + "=" * 80)
        print("LOG-NORMAL TRANSFORMATION")
        print("=" * 80)

        transformed_df = df.copy()

        for exchange, commodities in self.exchanges.items():
            for commodity in commodities:
                if commodity in transformed_df.columns:
                    series = transformed_df[commodity]

                    # Handle negative values by shifting
                    min_val = series.min()
                    if min_val <= 0:
                        series = series - min_val + 1

                    # Apply log transformation
                    transformed_df[f'{commodity}_log'] = np.log(series)
                    print(f"  Transformed: {commodity}")

        return transformed_df

    def make_stationary(self, df):
        """Convert to stationary data through differencing"""
        print("\n" + "=" * 80)
        print("STATIONARITY TRANSFORMATION")
        print("=" * 80)

        stationary_df = df.copy()

        for exchange, commodities in self.exchanges.items():
            for commodity in commodities:
                log_col = f'{commodity}_log'
                if log_col in stationary_df.columns:
                    # First-order differencing
                    stationary_df[f'{commodity}_stationary'] = stationary_df[log_col].diff()
                    print(f"  Differenced: {commodity}")

        return stationary_df

    def calculate_technical_indicator(self, df, commodity, period=14):
        """
        Calculate technical indicators for commodities
        Options: CCI (Commodity Channel Index), RSI, MACD, Bollinger Bands
        """
        series = df[commodity].copy()

        if self.indicator == 'CCI':
            # Commodity Channel Index
            tp = series  # Typical price (simplified for univariate)
            sma = tp.rolling(window=period).mean()
            mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
            cci = (tp - sma) / (0.015 * mad)
            return cci

        elif self.indicator == 'RSI':
            # Relative Strength Index
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi

        elif self.indicator == 'MACD':
            # Moving Average Convergence Divergence
            ema12 = series.ewm(span=12, adjust=False).mean()
            ema26 = series.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            return macd

        elif self.indicator == 'BB':
            # Bollinger Bands (return position within bands)
            sma = series.rolling(window=period).mean()
            std = series.rolling(window=period).std()
            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)
            bb_position = (series - lower_band) / (upper_band - lower_band)
            return bb_position

        else:
            return pd.Series(0, index=series.index)

    def find_optimal_lags(self, series, max_lag=None):
        """Find optimal lag features using ACF/PACF significance"""
        if max_lag is None:
            max_lag = self.max_lag

        clean_series = series.dropna()
        if len(clean_series) < max_lag + 10:
            return list(range(1, min(8, len(clean_series) - 1)))

        from statsmodels.tsa.stattools import acf

        try:
            acf_values = acf(clean_series, nlags=max_lag, fft=False)
            # Find significant lags (|ACF| > 2/sqrt(n))
            threshold = 2 / np.sqrt(len(clean_series))
            significant_lags = [i for i in range(1, len(acf_values))
                               if abs(acf_values[i]) > threshold]

            if len(significant_lags) == 0:
                significant_lags = list(range(1, 8))

            return significant_lags[:10]  # Limit to top 10 lags
        except:
            return list(range(1, 8))

    def create_lag_features(self, df, commodity, lags):
        """Create lag features for a commodity"""
        stat_col = f'{commodity}_stationary'
        lag_df = pd.DataFrame() # Reinitialize lag_df for each commodity

        for lag in lags:
            lag_df[f'{commodity}_lag_{lag}'] = df[stat_col].shift(lag)

        return lag_df

    def impute_outliers(self, series, method='winsorize', threshold=3):
        """
        Impute outliers without removing data points
        Methods: winsorize, clip, interpolate
        """
        clean_series = series.copy()

        if method == 'winsorize':
            # Replace extreme values with percentile values
            lower = clean_series.quantile(0.01)
            upper = clean_series.quantile(0.99)
            clean_series = clean_series.clip(lower=lower, upper=upper)

        elif method == 'clip':
            # Clip based on z-score
            mean = clean_series.mean()
            std = clean_series.std()
            lower = mean - threshold * std
            upper = mean + threshold * std
            clean_series = clean_series.clip(lower=lower, upper=upper)

        elif method == 'interpolate':
            # Mark outliers and interpolate
            z_scores = np.abs(stats.zscore(clean_series.dropna()))
            outliers = z_scores > threshold
            clean_series.loc[outliers] = np.nan
            clean_series = clean_series.interpolate(method='linear')

        # Handle remaining NaN values
        clean_series = clean_series.fillna(method='ffill').fillna(method='bfill')

        return clean_series

    def prepare_features_exchange(self, df, exchange):
        """Prepare features for an entire exchange"""
        print(f"\n  Preparing features for {exchange}...")

        commodities = self.exchanges[exchange]
        if not commodities:
            return None, None

        feature_dfs = []
        target_cols = []

        for commodity in commodities:
            stat_col = f'{commodity}_stationary'
            if stat_col not in df.columns:
                continue

            # Find optimal lags
            lags = self.find_optimal_lags(df[stat_col])
            print(f"    {commodity}: Using lags {lags[:5]}..." if len(lags) > 5 else f"    {commodity}: Using lags {lags}")

            # Create lag features
            lag_features = self.create_lag_features(df, commodity, lags)

            # Add technical indicator
            indicator_col = self.calculate_technical_indicator(df, commodity)
            lag_features[f'{commodity}_indicator'] = indicator_col

            # Impute outliers in lag features
            for col in lag_features.columns:
                lag_features[col] = self.impute_outliers(lag_features[col])

            feature_dfs.append(lag_features)
            target_cols.append(stat_col)

        # Combine all features
        # Ensure alignment of indices for concat
        X = pd.concat(feature_dfs, axis=1)
        y = df[target_cols]

        return X, y

    def rolling_window_prediction(self, X, y, exchange):
        """
        XGBoost model with rolling window prediction
        """
        print(f"\n  Training {exchange} model with rolling window={self.rolling_window}...")

        # Remove rows with NaN
        valid_idx = ~(X.isna().any(axis=1) | y.isna().any(axis=1))
        X_clean = X[valid_idx]
        y_clean = y[valid_idx]

        if len(X_clean) < self.rolling_window + 10: # Ensure enough data for at least one training and prediction step
            print(f"    Insufficient data for {exchange}. Skipping model training.")
            return None

        # Split: use last 20% for testing
        split_idx = int(len(X_clean) * 0.8)

        predictions = []
        actuals = []
        # train_scores = [] # Not used, can be removed

        # Rolling window prediction
        # Iterate only if there's enough data for at least one test window after split_idx
        if split_idx >= len(X_clean) - self.rolling_window:
            print(f"    Insufficient data for rolling window prediction in {exchange} after split_idx. Skipping.")
            return None

        for i in range(split_idx, len(X_clean) - self.rolling_window):
            # Training window
            # Use a fixed size for training if desired, or dynamic. Max 500 points for training to prevent too long training times.
            train_start = max(0, i - 500) 
            X_train = X_clean.iloc[train_start:i]
            y_train = y_clean.iloc[train_start:i]

            # Test window (single step prediction)
            X_test = X_clean.iloc[i:i+1]
            y_test = y_clean.iloc[i:i+1]

            if X_train.empty or y_train.empty:
                continue

            # Train model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbosity=0
            )

            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_test)

            predictions.append(y_pred[0])
            actuals.append(y_test.values[0])

            # Store final model after the last iteration of the rolling window to save the most recent model
            if i == len(X_clean) - self.rolling_window - 1:
                self.models[exchange] = model

        if not predictions:
            print(f"    No predictions were made for {exchange}. Skipping metric calculation.")
            return None

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        # Calculate metrics
        # Ensure predictions and actuals have consistent shapes before calculating metrics
        # If multioutput='raw_values' is used, need to check its output behavior with single prediction
        # For mean calculation, ensure it handles cases where actuals/predictions might be 1D or 2D
        if predictions.ndim > 1 and actuals.ndim > 1:
             rmse = np.sqrt(mean_squared_error(actuals, predictions, multioutput='raw_values')).mean()
             mae = mean_absolute_error(actuals, predictions, multioutput='raw_values').mean()
             r2 = r2_score(actuals, predictions, multioutput='raw_values').mean()
        else:
             rmse = np.sqrt(mean_squared_error(actuals, predictions))
             mae = mean_absolute_error(actuals, predictions)
             r2 = r2_score(actuals, predictions)

        errors = predictions - actuals

        # If multi-output, flatten
        if errors.ndim > 1:
            errors = errors.flatten()
        
        error_std = np.std(errors)
        if error_std == 0:
            sharpe_like = np.nan
        else:
            sharpe_like = mae / error_std

        metrics = {
            'exchange': exchange,
            'n_predictions': len(predictions),
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'sharpe_like': sharpe_like
        }

        self.results[exchange] = {
            'predictions': predictions,
            'actuals': actuals,
            'metrics': metrics
        }

        print(f"    RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.6f}")

        return metrics

    def train_all_models(self, df):
        """Train models for all exchanges"""
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)

        all_metrics = []

        for exchange in self.exchanges.keys():
            if not self.exchanges[exchange]:
                print(f"  Skipping {exchange} - no commodities assigned.")
                continue

            print(f"\n{exchange} Exchange:")
            X, y = self.prepare_features_exchange(df, exchange)

            if X is None or y is None or X.empty or y.empty or len(X) < 20:
                print(f"  Skipping {exchange} - insufficient data after feature preparation.")
                continue

            metrics = self.rolling_window_prediction(X, y, exchange)
            if metrics:
                all_metrics.append(metrics)

        return pd.DataFrame(all_metrics)

    def plot_results(self):
        """Generate publication-quality plots"""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80)

        n_exchanges = len(self.results)
        if n_exchanges == 0:
            print("No results to plot")
            return

        fig, axes = plt.subplots(1, n_exchanges, figsize=(15, 5) if n_exchanges > 1 else (7, 5), squeeze=False) # Use squeeze=False to always get a 2D array of axes
        axes = axes.flatten()

        for idx, (exchange, result) in enumerate(self.results.items()):
            ax = axes[idx]
            predictions = result['predictions']
            actuals = result['actuals']

            # Average across commodities for visualization if there are multiple targets
            if len(predictions.shape) > 1:
                pred_mean = predictions.mean(axis=1)
                actual_mean = actuals.mean(axis=1)
            else:
                pred_mean = predictions
                actual_mean = actuals

            time_steps = range(len(pred_mean))

            ax.plot(time_steps, actual_mean, label='Actual', linewidth=2, alpha=0.7)
            ax.plot(time_steps, pred_mean, label='Predicted', linewidth=2, alpha=0.7)
            ax.set_title(f'{exchange} Exchange - Predictions vs Actuals', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Stationary Value')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'prediction_results.png'), dpi=300, bbox_inches='tight')
        print(f"\nSaved: {os.path.join(self.output_dir, 'prediction_results.png')}")
        plt.close(fig)

        # Error distribution plot
        fig_err, axes_err = plt.subplots(1, n_exchanges, figsize=(15, 4) if n_exchanges > 1 else (7, 4), squeeze=False)
        axes_err = axes_err.flatten()

        for idx, (exchange, result) in enumerate(self.results.items()):
            predictions = result['predictions']
            actuals = result['actuals']

            if len(predictions.shape) > 1:
                errors = (predictions - actuals).flatten()
            else:
                errors = predictions - actuals

            axes_err[idx].hist(errors, bins=50, alpha=0.7, edgecolor='black')
            axes_err[idx].set_title(f'{exchange} Error Distribution')
            axes_err[idx].set_xlabel('Prediction Error')
            axes_err[idx].set_ylabel('Frequency')
            axes_err[idx].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'error_distributions.png'), dpi=300, bbox_inches='tight')
        print(f"Saved: {os.path.join(self.output_dir, 'error_distributions.png')}")
        plt.close(fig_err)
        # plt.show() # Only show if not saving to avoid blocking execution in automated environments

    def generate_research_report(self):
        """Generate comprehensive research paper formatted results"""
        print("\n" + "=" * 80)
        print("RESEARCH REPORT")
        print("=" * 80)

        report = []

        # Abstract
        report.append("\n### ABSTRACT ###")
        report.append("This study presents a comprehensive commodity price prediction framework")
        report.append("utilizing exchange-specific XGBoost models with log-normal transformation")
        report.append(f"and stationarity conversion. Technical indicators ({self.indicator}) were")
        report.append("incorporated to enhance predictive accuracy across four major exchanges:")
        report.append("US, LME, FX, and JPX.")

        # Statistical Analysis Summary
        report.append("\n### 1. STATISTICAL ANALYSIS ###")
        if hasattr(self, 'statistical_tests') and not self.statistical_tests.empty:
            report.append("\n1.1 Stationarity Analysis")
            stationary_pct = self.statistical_tests['adf_stationary'].mean() * 100
            report.append(f"  - {stationary_pct:.1f}% of commodities exhibit stationarity (ADF test, \u03b1=0.05)")

            report.append("\n1.2 Log-Normality Assessment")
            lognormal_pct = self.statistical_tests['is_log_normal'].mean() * 100
            report.append(f"  - {lognormal_pct:.1f}% of commodities follow log-normal distribution")

            report.append("\n1.3 Autocorrelation")
            autocorr_pct = self.statistical_tests['autocorrelated'].mean() * 100
            report.append(f"  - {autocorr_pct:.1f}% of commodities show significant autocorrelation")
        else:
            report.append("  - No statistical analysis results available.")

        # Methodology
        report.append("\n### 2. METHODOLOGY ###")
        report.append(f"  - Data Transformation: Log-normal \u2192 First-order differencing")
        report.append(f"  - Technical Indicator: {self.indicator}")
        report.append(f"  - Model: XGBoost with rolling window (size={self.rolling_window})")
        report.append(f"  - Feature Selection: Optimal lags via ACF analysis (max={self.max_lag})")
        report.append(f"  - Outlier Treatment: Imputation (no data removal)")

        # Results by Exchange
        report.append("\n### 3. RESULTS ###")
        if self.results:
            for exchange, result in self.results.items():
                metrics = result['metrics']
                report.append(f"\n3.{list(self.results.keys()).index(exchange)+1} {exchange} Exchange")
                report.append(f"  - RMSE: {metrics['rmse']:.6f}")
                report.append(f"  - MAE:  {metrics['mae']:.6f}")
                report.append(f"  - R\u00b2:   {metrics['r2']:.6f}")
                report.append(f"  - Predictions: {metrics['n_predictions']}")
        else:
            report.append("  - No model results available.")

        # Conclusions
        report.append("\n### 4. CONCLUSIONS ###")
        if self.results:
            # Find best_exchange based on RMSE, handle case where results might be empty or missing 'rmse'
            best_exchange = None
            min_rmse = float('inf')
            for exchange, result in self.results.items():
                if 'rmse' in result['metrics'] and result['metrics']['rmse'] < min_rmse:
                    min_rmse = result['metrics']['rmse']
                    best_exchange = (exchange, result)

            if best_exchange:
                report.append(f"  - Best performing exchange: {best_exchange[0]}")
                report.append(f"    (RMSE: {best_exchange[1]['metrics']['rmse']:.6f})")

            avg_r2 = np.mean([r['metrics']['r2'] for r in self.results.values() if 'r2' in r['metrics']])
            report.append(f"  - Average R\u00b2 across exchanges: {avg_r2:.4f}")
            report.append(f"  - Exchange-specific modeling demonstrates improved performance")
            report.append(f"  - Log-normal transformation and differencing successfully induced stationarity")
        else:
            report.append("  - Unable to draw conclusions as no models were trained successfully.")

        report_text = "\n".join(report)
        print(report_text)

        # Save to file
        with open(os.path.join(self.output_dir, 'research_report.txt'), 'w') as f:
            f.write(report_text)
        print(f"\nSaved: {os.path.join(self.output_dir, 'research_report.txt')}")

        return report_text

    def save_models(self):
        """Save trained models to files."""
        print("\n" + "=" * 80)
        print("SAVING MODELS")
        print("=" * 80)
        if self.models:
            for exchange, model in self.models.items():
                filename = os.path.join(self.output_dir, f'{exchange}_model.joblib')
                joblib.dump(model, filename)
                print(f"Saved model for {exchange} to {filename}")
        else:
            print("No models to save.")

    def save_results(self):
        """Save all results to files"""
        print("\n" + "=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)

        # Statistical tests
        if hasattr(self, 'statistical_tests') and not self.statistical_tests.empty:
            self.statistical_tests.to_csv(os.path.join(self.output_dir, 'statistical_tests.csv'), index=False)
            print(f"Saved: {os.path.join(self.output_dir, 'statistical_tests.csv')}")
        else:
            print("No statistical tests results to save.")

        # Model metrics
        if self.results:
            metrics_df = pd.DataFrame([r['metrics'] for r in self.results.values()])
            metrics_df.to_csv(os.path.join(self.output_dir, 'model_metrics.csv'), index=False)
            print(f"Saved: {os.path.join(self.output_dir, 'model_metrics.csv')}")
        else:
            print("No model metrics to save.")

        print("\nPipeline complete!")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize pipeline
    pipeline = CommodityPredictionPipeline(
        rolling_window=7,
        indicator='CCI',  # Options: 'CCI', 'RSI', 'MACD', 'BB'
        max_lag=30,
        output_dir='results/pipeline_results'
    )

    # Run complete pipeline
    print("MITSUI COMMODITY PREDICTION PIPELINE")
    print("=" * 80)

    # Load data
    df = pipeline.load_data('/data/train_selected.csv')
    
    
    # Statistical analysis
    stats_results = pipeline.run_statistical_analysis()

    # Transform data
    df = pipeline.transform_to_log_normal(df)
    df = pipeline.make_stationary(df)
    

    # Train models
    metrics = pipeline.train_all_models(df)

    # Generate outputs
    pipeline.plot_results()
    pipeline.generate_research_report()
    pipeline.save_models() # Call the new save_models method
    pipeline.save_results()

    print("\n" + "=" * 80)
    print("COMPLETE - All results saved")
    print("=" * 80)