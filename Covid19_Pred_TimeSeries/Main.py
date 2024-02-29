#%%
# -- 1.Setup --
import os
import datetime

import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from time_series_helper import WindowGenerator
from sklearn.metrics import mean_absolute_error

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False   # Turn off the grid

#%%
# -- 2. Data Loading --
CSV_PATH = os.path.join(os.getcwd(), 'cases_malaysia_covid_FULL.csv')
df = pd.read_csv(CSV_PATH)

#%%
# -- 3. Data Exploratory --
df.info()
df.head()

#%%
df.describe().transpose()

# %%
# Check for duplicate
print(f"Number of duplicate: {df.duplicated().sum()}")

# --- FOUNDING --- 
# 1. Need to change cases_new column type
# 2. .info() shows presence of NULL value in cases_new column, but in reality dataset shows there are gap values
# 3. Last 8 columns contain NULL values, may due to there are not yet this clusters in the begining of Covid-19.
# 4. No duplicates found

#%%
# -- 4. Data Preprocessing --

# Deal with cases_new column
df['cases_new'] = pd.to_numeric(df['cases_new'], errors='coerce')             # Change the new_casses object features type and check for NULL values again
df['cases_new'].isna().sum()                                                  # Confirm on the NULL values
df['cases_new'] = df['cases_new'].interpolate(method='polynomial', order=2)   # Replace NULL using interpolation method
df['cases_new'] = df['cases_new'].astype('int64')                             # Round of to int

# %%
# -- 5.Data Visualization and Cleaning --

# Extract out the date column and convert it into datetime format
date_time = pd.to_datetime(df.pop('date'), format='%d/%m/%Y')

# Evoluiton of cases_news over time
plot_cols = ['cases_new']
plot_features = df[plot_cols]
plot_features.index = date_time
_ = plot_features.plot(subplots=True)
plt.show()

plot_features = df[plot_cols][:480]
plot_features.index = date_time[:480]
_ = plot_features.plot(subplots=True)
plt.show()
# The patterns here show that the new cases are increasing at early 2022 and the peak are around Feb 2021

# Outliers
df.boxplot()  # Extreme values, so might ignore it
plt.show()

# %%
# -- 6. Data Splitting and Normalization --
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
df = df[plot_cols]
train_df = df[0:int(n*0.7)]
val_df = df[int(n*0.7):int(n*0.9)]
test_df = df[int(n*0.9):]

num_features = df.shape[1]

# Normalization
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

# Perform a violin plot to look at the distribution of the features
df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')
plt.figure(figsize=(12, 6))
ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
_ = ax.set_xticklabels(df.keys(), rotation=90)
# %%
# -- 7. Data Window --

# Single-step window
single_window = WindowGenerator(
    input_width=30, label_width=30, shift=1,
    train_df=train_df, val_df=val_df, test_df=test_df,
    label_columns=['cases_new'],
    batch_size=128)

# Multi-step window
OUT_STEPS = 30
multi_window = WindowGenerator(input_width=30, label_width=OUT_STEPS, shift=OUT_STEPS,
                               train_df=train_df, val_df=val_df, test_df=test_df,
                               label_columns=['cases_new'])

single_window.plot(plot_col='cases_new')
plt.show()
print(single_window)
multi_window.plot(plot_col='cases_new')
plt.show()
# %%
# -- 8a. Model Development (Single-step Model) --

# Single-step model
lstm_model = tf.keras.Sequential()
lstm_model.add(tf.keras.layers.LSTM(32, return_sequences=True))
lstm_model.add(tf.keras.layers.Dense(1))

# Define a function to compile and train the model
MAX_EPOCHS = 50
logpath = os.path.join(os.getcwd(), 'tensorboard_log', datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
tb = tf.keras.callbacks.TensorBoard(logpath)

def compile_and_fit(model, window, epochs=MAX_EPOCHS, patience=12):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping, tb])
  return history

#%%
# -- 9a. Model Training (Single-step Model) --

history_single = compile_and_fit(lstm_model, single_window)

#%%
# -- 10a. Model Evaluation (Single-step Model) --

single_window.plot(lstm_model, plot_col='cases_new')
# %%
# -- 8b. Model Development (Multi-step Model)

# Multi-step Model
multi_lstm_model = tf.keras.Sequential([
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(OUT_STEPS*1,
                          kernel_initializer=tf.initializers.zeros()),
    tf.keras.layers.Reshape([OUT_STEPS, 1])
])

#%%
# -- 9b. Model Training (Multi-step Model) --

history_multi = compile_and_fit(multi_lstm_model, multi_window)

#%% 
# -- 10b. Model Evaluation (Multi-step Model) --

multi_window.plot(multi_lstm_model, plot_col='cases_new')

#%%
# -- 11. Performance Evaluation --

print(history_single.history.keys())    # look for keys to extract for calculate mape

# Calculate MAPE
def calculate_mape(model_name, model_history, true_values):
     sum_abs_values = np.sum(np.abs(true_values))
     mae = model_history.history['mean_absolute_error'][-1]     # access the latest values
     mape = (mae / sum_abs_values) * 100
     print(f"MAPE for {model_name} on validaiton set: {mape:.2f}%")

calculate_mape('Single-step Model', history_single, single_window.val_df['cases_new'])
calculate_mape('Multi-step Model', history_multi, multi_window.val_df['cases_new'])

# Model Architecture
# tf.keras.utils.plot_model(lstm_model)
# tf.keras.utils.plot_model(multi_lstm_model)
