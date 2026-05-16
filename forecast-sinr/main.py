import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

from window_generator import WindowGenerator
from tools import Plotter
from tools import Normalization

from tensorflow.python.keras.mixed_precision import policy as mp_policy
policy = mp_policy.Policy('mixed_float16')
mp_policy.set_global_policy(policy)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

total_df = pd.read_csv("/home/luis_massagardi/5G-air-simulator/TOOLS/SINR_OneUe_1.4Mhz_10min.txt", names=['sinr', 'time'])

df = total_df[['sinr']]

# Plotter.plot_df(df=df, total_df=total_df)

column_indices = {name: i for i, name in enumerate(df.columns)} 
n = len(df)
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1] 

SCALING_WINDOW_SIZE = 100

train_df = Normalization.rolling_z_score(train_df, SCALING_WINDOW_SIZE)
val_df = Normalization.rolling_z_score(val_df, SCALING_WINDOW_SIZE, prev_df=train_df)
test_df = Normalization.rolling_z_score(test_df, SCALING_WINDOW_SIZE, prev_df=val_df)
train_df = train_df.dropna()

Plotter.plot_violin(train_df)

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=162, activation='tanh', recurrent_dropout=0, return_sequences=False),
    tf.keras.layers.Dense(units=1, dtype='float32')
])

OUT_STEPS = 10

IN_STEPS = 20
    
MAX_EPOCHS = 50

multi_window = WindowGenerator(input_width=IN_STEPS, label_width=OUT_STEPS,
                               shift=OUT_STEPS, train_df=train_df, 
                               val_df=val_df, test_df=test_df,
                               label_columns=['sinr'])

def compile_and_fit(model, window, patience=10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

    history = model.fit(
        window.train,
        epochs=MAX_EPOCHS, 
        validation_data=multi_window.val,
        callbacks=[early_stopping]
    )
    return history

history = compile_and_fit(lstm_model, multi_window)

val_performance = {}
performance = {}
val_performance['LSTM'] = lstm_model.evaluate(multi_window.val, return_dict=True)
performance['LSTM'] = lstm_model.evaluate(multi_window.test, verbose=0, return_dict=True)

lstm_model.save('model_sinr_lstm_10min.keras')

WindowGenerator.plot = Plotter.plot_window_performance

multi_window.plot(lstm_model)

plt.savefig('model_sinr_lstm_plot_performance.png')