import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv', nrows=1000)

#-Atsirenkame reikiamus duomenis
selected_features = ['Product Category', 'Product Price', 'Quantity', 'Purchase Date']
df_selected = df[selected_features]

#-Konvertuojame 'Purchase Date' to datetime formatą
df_selected['Purchase Date'] = pd.to_datetime(df_selected['Purchase Date'])
df_selected['Year'] = df_selected['Purchase Date'].dt.year
df_selected['Month'] = df_selected['Purchase Date'].dt.month
df_selected['Day'] = df_selected['Purchase Date'].dt.day
df_selected['Hour'] = df_selected['Purchase Date'].dt.hour
df_selected['Minute'] = df_selected['Purchase Date'].dt.minute

#-Sugrupuojame pirkimus pagal mėn. ir metus.
df_grouped = df_selected.groupby(['Year', 'Month']).agg({'Quantity': 'sum'}).reset_index()

#-Duomenų standartizavimas
scaler = MinMaxScaler(feature_range=(0, 1))
df_grouped['Scaled Quantity'] = scaler.fit_transform(df_grouped['Quantity'].values.reshape(-1, 1))

#-Paruošimas
look_back = 6
X, Y = [], []
for i in range(look_back, len(df_grouped)):
    X.append(df_grouped.iloc[i - look_back:i]['Scaled Quantity'].values)
    Y.append(df_grouped.iloc[i]['Scaled Quantity'])

X, Y = np.array(X), np.array(Y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

#-Gru modelio kūrimas
model = Sequential([
    GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    GRU(units=64),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

#-Modelio treniravimas
model.fit(X_train, Y_train, epochs=11, batch_size=32, validation_data=(X_test, Y_test), verbose=1)

#-Spėjimai
predictions = model.predict(X_test)

#-Duomenų atstatymas į pradines reikšmes
predictions = scaler.inverse_transform(predictions)
Y_test = scaler.inverse_transform(Y_test.reshape(-1, 1))

#-Skaičiuojame RMSE
rmse = np.sqrt(mean_squared_error(Y_test, predictions))
print("RMSE:", rmse)

#-Skaičiuojame MAE
mae = mean_absolute_error(Y_test, predictions)
print("MAE:", mae)

#-Skaičiuojame R2 score
r2 = r2_score(Y_test, predictions)
print("R2:", r2)

#-Vizualizacija
plt.figure(figsize=(10, 6))
plt.plot(Y_test, label='Actual Quantity', color='blue')
plt.plot(predictions, label='Predicted Quantity', color='orange')
plt.xlabel('Imties indeksas')
plt.ylabel('Kiekis')
plt.title('Faktinės ir prognozuotos pirkimų kiekis')
plt.legend()
plt.show()
