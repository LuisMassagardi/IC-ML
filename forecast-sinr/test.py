import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from window_generator import WindowGenerator
from tools import Plotter, Normalization

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

model = tf.keras.models.load_model('best_sinr_lstm.keras')

total_df = pd.read_csv("/home/luis_massagardi/5G-air-simulator/TOOLS/SINR_OneUe_1.4Mhz_10min.txt", names=['sinr', 'time'])
df = total_df[['sinr']]

Plotter.plot_df(df=df, total_df=total_df)

n = len(df)
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

SCALING_WINDOW_SIZE = 100
val_df_norm = Normalization.rolling_z_score(val_df, SCALING_WINDOW_SIZE).dropna()
test_df_norm = Normalization.rolling_z_score(test_df, SCALING_WINDOW_SIZE, prev_df=val_df)

Plotter.plot_violin(val_df_norm)

val_window = WindowGenerator(
    input_width=20, label_width=10, shift=10,
    train_df=val_df_norm, val_df=val_df_norm, test_df=val_df_norm,
    label_columns=['sinr']
)

test_window = WindowGenerator(
    input_width=20, 
    label_width=10, 
    shift=10,
    train_df=test_df_norm, val_df=test_df_norm, test_df=test_df_norm,
    label_columns=['sinr']
)

val_performance = {}
test_performance = {}
val_performance['LSTM'] = model.evaluate(val_window.val, return_dict=True, verbose=1)
test_performance['LSTM'] = model.evaluate(test_window.test, return_dict=True, verbose=1)

WindowGenerator.plot = Plotter.plot_window_performance
test_window.plot(model)

plt.savefig('Test_performance.png')

Plotter.plot_actual_vs_predictions(test_window, model, df, manual_start=0, window_size=50)

plt.savefig('Test_actual_vs_predicted.png')

Plotter.plot_prediction_error_scatter(test_window, model, df)

plt.savefig('Test_Scatter.png')

#  manual_start=400, window_size=80