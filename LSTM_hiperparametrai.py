import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from scipy.stats import randint as sp_randint

# Duomenų įkėlimas
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv', nrows=20000)

# Paruošimas
df['Purchase Date'] = df['Purchase Date'].apply(lambda x: pd.to_datetime(x).timestamp())
X = df[['Purchase Date', 'Product Category', 'Product Price']]
y = df['Quantity']

# Kategorinių kodavimas
categorical_features = ['Product Category']
trans = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
X_encoded = trans.fit_transform(X)

# Duomenų standartizavimas
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_encoded)

# Duomenų padalinimas
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modelis
def build_model(optimizer='adam', units=64):
    model = Sequential()
    model.add(Dense(units, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.5))
    model.add(Dense(units // 2, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mse'])
    return model


model = KerasRegressor(model=build_model, verbose=0)


# Hiperparametrai
param_dist = {
    'model__optimizer': ['adam', 'rmsprop'],
    'model__units': sp_randint(32, 256),
    'epochs': sp_randint(10, 100),
    'batch_size': sp_randint(10, 100)
}

# RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, cv=3)

# Paieška
random_search_result = random_search.fit(X_train, y_train)

# Geriausi parametrai
print("Geriausias rezultatas: %f naudojant %s" % (random_search_result.best_score_, random_search_result.best_params_))