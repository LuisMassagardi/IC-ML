import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt

class Plotter():
    @staticmethod
    def plot_df(df, total_df):
        n = len(df)
        half_df = total_df[0:int(n*1)]
        plt.figure(figsize=(12, 6))
        plt.scatter(half_df['time'], half_df['sinr'], s=10)
        plt.title('OneUE SINR vs Time')
        plt.xlabel('Time (s)')
        plt.ylabel('SINR (dB)')
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.savefig('Final_test_dataset.png')
    
    @staticmethod
    def plot_violin(scaled_df):
        df_melted = scaled_df.melt(var_name='Column', value_name='Normalized Z-Score')
        plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='Column', y='Normalized Z-Score', data=df_melted)
        labels = scaled_df.columns
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=90)
        plt.axhline(0, color='red', linestyle='--', alpha=0.6)
        plt.grid(True, which='both', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig('tst-2')

    def plot_window_performance(self, model, plot_col='sinr'):
        inputs, labels = self.example
        plot_col_index = self.column_indices[plot_col]
        
        predictions = model(inputs)
        
        if len(predictions.shape) == 3:
            errors = np.mean(np.abs(predictions - labels), axis=(1, 2))
        else:
            errors = np.mean(np.abs(predictions - labels), axis=1)
            
        sorted_indices = np.argsort(errors) 
    
        best_idx = sorted_indices[0]
        worst_idx = sorted_indices[-1]
        mid_idx = sorted_indices[len(sorted_indices) // 2]
    
        ranked_indices = [best_idx, mid_idx, worst_idx]
        titles = ['Lowest Error', 'Median Error', 'Highest Error']

        plt.figure(figsize=(12, 10))
        for i, n in enumerate(ranked_indices):
            plt.subplot(3, 1, i + 1)
            plt.ylabel(f'{plot_col} (norm)')
            plt.title(f"{titles[i]} - Window Index: {n} - MAE: {errors[n]:.4f}")

            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                    label='Inputs', marker='.', zorder=-10)

            label_col_index = self.label_columns_indices.get(plot_col, plot_col_index)
            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                    edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if len(predictions.shape) == 2:
                plot_preds = predictions[n, label_col_index]
            else:
                plot_preds = predictions[n, :, label_col_index]

            plt.scatter(self.label_indices, plot_preds,
                        marker='X', edgecolors='k', label='Predictions',
                        c='#ff7f0e', s=64)
        
            if i == 0:
                plt.legend()
            
        plt.tight_layout()
    
    @staticmethod
    def plot_actual_vs_predictions(window, model, df, title="Actual vs Predicted"):
        all_predictions_norm = []
        all_actuals_norm = []
        
        for inputs, labels in window.test:
            predictions = model(inputs)
            all_predictions_norm.extend(predictions[:, 0, 0].numpy())
            all_actuals_norm.extend(labels[:, 0, 0].numpy())
        

        rolling_mean = df['sinr'].rolling(window=100).mean()
        rolling_std = df['sinr'].rolling(window=100).std() + 1e-8
    
        offset = len(df) - len(all_actuals_norm)
        stats_mean = rolling_mean.iloc[offset:].values
        stats_std = rolling_std.iloc[offset:].values

        all_actuals_db = (np.array(all_actuals_norm) * stats_std) + stats_mean
        all_predictions_db = (np.array(all_predictions_norm) * stats_std) + stats_mean

        errors = np.abs(all_actuals_db - all_predictions_db)
        max_error_idx = np.argmax(errors)

        window_size = 30
        start_idx = max(0, max_error_idx - (window_size // 2))
        end_idx = min(len(all_actuals_db), start_idx + window_size)
    
        if end_idx == len(all_actuals_db):
            start_idx = max(0, end_idx - window_size)

        plot_actuals = all_actuals_db[start_idx:end_idx]
        plot_preds = all_predictions_db[start_idx:end_idx]
            
        plt.figure(figsize=(15, 6))
        plt.plot(plot_actuals, label='Actual SINR', marker='.', color='blue', alpha=0.6, linewidth=1)
        plt.plot(plot_preds, label='Predicted SINR', marker='.', color='orange', alpha=0.8, linestyle='--')
        plt.title(title)
        plt.xlabel('Timesteps (ms)')
        plt.ylabel('SINR (dB)')
        plt.legend()
        plt.grid(True, alpha=0.4)
        plt.tight_layout()
            
class Normalization:
    @staticmethod
    def rolling_z_score(df, window, prev_df=None):
        if prev_df is not None:
            context = prev_df.tail(window - 1)
            full_df = pd.concat([context, df])
        else:
            full_df = df
        
        rolling_mean = full_df.rolling(window=window).mean()
        rolling_std = full_df.rolling(window=window).std() + 1e-8 #const avoids division by zero

        scaled_df = (full_df - rolling_mean) / rolling_std 

        if prev_df is not None:
            return scaled_df.iloc[window - 1:]
        
        return scaled_df