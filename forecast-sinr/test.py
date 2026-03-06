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

total_df = pd.read_csv("/home/luis_massagardi/5G-air-simulator/TOOLS/SINR_OneUe_1.4Mhz_1min_2.txt", names=['sinr', 'time'])
df = total_df[['sinr']]

Plotter.plot_df(df=df, total_df=total_df)

## Data normalization
SCALING_WINDOW_SIZE = 100

test_df = Normalization.rolling_z_score(df, SCALING_WINDOW_SIZE)
test_df = test_df.dropna()

Plotter.plot_violin(test_df)

test_window = WindowGenerator(
    input_width=20, 
    label_width=10, 
    shift=10,
    train_df=test_df, val_df=test_df, test_df=test_df,
    label_columns=['sinr']
)

performance = model.evaluate(test_window.test, verbose=1)

WindowGenerator.plot = Plotter.plot_window_performance
test_window.plot(model)

plt.savefig('Final_plot_performance.png')

Plotter.plot_actual_vs_predictions(test_window, model, df)

plt.savefig('Final_actual_vs_predicted.png')


