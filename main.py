import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from window_generator import WindowGenerator
from tools import Plotter
from tools import Normalization

### Limit tensorflow gpu usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

total_df = pd.read_csv("/home/luis_massagardi/5G-air-simulator/TOOLS/SINR_OneUe_1.4Mhz_1min_2.txt", names=['sinr', 'time'])

df = total_df[['sinr']]

Plotter.plot_df(df=df, total_df=total_df)

### Split the data
column_indices = {name: i for i, name in enumerate(df.columns)} # turns column names to indexes
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1] # number of collumns (features in ML)

## Data normalization
SCALING_WINDOW_SIZE = 100

train_df = Normalization.rolling_z_score(train_df, SCALING_WINDOW_SIZE)
val_df = Normalization.rolling_z_score(val_df, SCALING_WINDOW_SIZE, prev_df=train_df)
test_df = Normalization.rolling_z_score(test_df, SCALING_WINDOW_SIZE, prev_df=val_df)
train_df = train_df.dropna()

Plotter.plot_violin(train_df)

multi_val_performance = {}
multi_performance = {}

OUT_STEPS = 5

multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units].
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps*features].
    tf.keras.layers.Dense(OUT_STEPS*num_features,
                          kernel_initializer=tf.initializers.zeros()),
    # Shape => [batch, out_steps, features].
    tf.keras.layers.Reshape([OUT_STEPS, num_features])
])

MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

multi_window = WindowGenerator(input_width=20, label_width=OUT_STEPS,
                               shift=OUT_STEPS, train_df=train_df, 
                               val_df=val_df, test_df=test_df,
                               label_columns=['sinr'])

history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val, return_dict=True)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test, verbose=1, return_dict=True)

WindowGenerator.plot = Plotter.plot_window_performance
multi_window.plot(multi_lstm_model)
plt.savefig('Window_Performance_1min-2.png')