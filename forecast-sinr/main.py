import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import keras_tuner as kt

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

total_df = pd.read_csv("/home/luis_massagardi/5G-air-simulator/TOOLS/SINR_OneUe_1.4Mhz_15min_1.txt", names=['sinr', 'time'])

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

OUT_STEPS = 10

IN_STEPS = 20

def model_builder(hp):
    model = tf.keras.Sequential()
    
    hp_units = hp.Int('units', min_value=16, max_value=256, step=16) # Tune the number of LSTM units
    
    hp_activation = hp.Choice('activation', values=['tanh', 'relu', 'selu']) # Tune the activation function
    
    model.add(tf.keras.layers.LSTM(units=hp_units, 
                                   activation=hp_activation,
                                   return_sequences=False))
    
    hp_dropout = hp.Float('dropout', min_value=0.0, max_value=0.5, step=0.1) # Tune Dropout rate
    model.add(tf.keras.layers.Dropout(hp_dropout))
    
    model.add(tf.keras.layers.Dense(OUT_STEPS * num_features,
                                    kernel_initializer=tf.initializers.zeros()))
    model.add(tf.keras.layers.Reshape([OUT_STEPS, num_features]))

    hp_lr = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4]) # Tune the Learning Rate

    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_lr),
        metrics=[tf.keras.metrics.MeanAbsoluteError()]
    )
    return model
    
MAX_EPOCHS = 50

multi_window = WindowGenerator(input_width=IN_STEPS, label_width=OUT_STEPS,
                               shift=OUT_STEPS, train_df=train_df, 
                               val_df=val_df, test_df=test_df,
                               label_columns=['sinr'])

# Initialize the Tuner (Bayesian Optimization is efficient for LSTMs)
tuner = kt.BayesianOptimization(
    model_builder,
    objective='val_loss',
    max_trials=50,            # How many different models to test
    executions_per_trial=1,    # How many times to train each model (for stability)
    directory='keras_tuner_dir',
    project_name='lstm_sinr_optimization',
    overwrite = True,
)

# Early stopping to keep the search from taking forever
callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Start the search process
tuner.search(
    multi_window.train,
    epochs=MAX_EPOCHS,
    validation_data=multi_window.val,
    callbacks=[callbacks],
)

top_3_hps = tuner.get_best_hyperparameters(num_trials=3)
print("\nTop 3 Models Found:")
for i, hp in enumerate(top_3_hps):
    print(f"Rank {i+1}: Units={hp.get('units')}, Activation={hp.get('activation')}, LR={hp.get('learning_rate')}, Dropout={hp.get('dropout')}")

best_hps = top_3_hps[0]

best_model = tuner.hypermodel.build(best_hps)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')

history = best_model.fit(
    multi_window.train,
    epochs=75, # You can increase this for the final model
    validation_data=multi_window.val,
    callbacks=[early_stopping]
)

best_model.save('best_sinr_lstm.keras')

WindowGenerator.plot = Plotter.plot_window_performance

multi_window.plot(best_model)

plt.savefig('best_model_plot_performance.png')