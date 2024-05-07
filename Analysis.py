import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import seaborn as sns

#Duomenu valymas ir paruosimas naudojimui
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# print(df.describe())

#Tikriname eiluciu skaiciu
# print(len(df))

# -Tikriname NaN reiksmes
# print(df.isnull().sum())

# -sutvarkome returns skilti, pasauliname NaN reiksmes
df['Returns'] = df['Returns'].fillna(0).astype(int)

# -konvertuojame pirkimo data to datetime
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
df['Purchase Year'] = df['Purchase Date'].dt.year
df['Purchase Month'] = df['Purchase Date'].dt.month_name()

df[['Purchase Date','Purchase Year','Purchase Month']]


#Analizuojame duomenis

# -Analizuojame duomenis pagal lyti(skaiciuojame pirkimu vidurki, )

pirkimai_pagal_lyti = df.groupby('Gender')['Total Purchase Amount'].mean()
# print(pirkimai_pagal_lyti)

mokejimo_metodas_pagal_lyti = df.groupby('Gender')['Payment Method'].value_counts()
# print(mokejimo_metodas_pagal_lyti)

#klientu amziaus grupes

jauniausias_klientas = df['Age'].min()
# print(f'Jauniausias klientas: {jauniausias_klientas}')
                                                         # suzinome tiek vyriausio, tiek jauniausio kliento amziu.
vyriausias_klientas = df['Age'].max()
# print(f'Vyriausias klientas: {vyriausias_klientas}')

df['amzius_iki_25'] = df[df['Age'] < 25].shape[0]
# print(f'Klientai jaunesni nei 25 m.: {amzius_iki_25}')

df['klientai_nuo_25_iki_50'] = df[(df['Age'] > 25) & (df['Age'] < 50)].shape[0]
# print(f'Klientai nuo 25 m. iki 50 m.: {klientai_nuo_25_iki_50}')

df['amzius_nuo_50'] = df[(df['Age'] > 50)].shape[0]
# print(f'Klientai nuo 50 m.: {amzius_nuo_50}')


#----------------------------------------------------------------
amzius_iki_25 = df[df['Age'] < 25].shape[0]
klientai_nuo_25_iki_50 = df[(df['Age'] > 25) & (df['Age'] < 50)].shape[0]
amzius_nuo_50 = df[(df['Age'] > 50)].shape[0]


age_groups = ['Klientai iki 25 m.', 'Klientai nuo 25 m. iki 50 m.', 'Klientai nuo 50m.']
quantity = [amzius_iki_25, klientai_nuo_25_iki_50, amzius_nuo_50]

features = pd.DataFrame({'Age groups': age_groups, 'Quantity': quantity})

plt.figure(figsize=(10, 8))
sns.barplot(data=features, x='Age groups', y='Quantity', palette='husl')
plt.xlabel('Amžiaus grupės', fontsize=13)
plt.ylabel('Klientų kiekis', fontsize=13)
plt.title('Klientų kiekis pagal amžiaus grupę', fontsize=15)
# plt.show()

#----------------------------------------------------------------

# pagal klientu lyti, bendros sumos isleistu pinigu pasiskirstymas

pirkimai_pagal_lyti = df.groupby(['Gender'])['Total Purchase Amount'].sum()    ##### pasitart kaaip
# print(pirkimai_pagal_lyti)

#___________________________________________________________________________________________________

##PARDAVIMAI PAGAL PREKIU KATEGORIJAS

pirmas_uzsakymo_data = df['Purchase Date'].min()
print(f'Pirmas uzsakymas atliktas: {pirmas_uzsakymo_data}')
paskutinis_uzsakymas = df['Purchase Date'].max()
print(f'Paskutinio uzsakymo data: {paskutinis_uzsakymas}')
prekiu_kategorijos = df['Product Category'].unique()
# print(prekiu_kategorijos)


##pardavimai pagal kategorijas per metus
df['Year'] = df['Purchase Date'].dt.year

prekiu_pardavimai_metams = df.groupby(['Year', 'Product Category'])['Total Purchase Amount'].sum()
# print(f'Prekiu pardavimai metams pagal kategorijas ir bendra pirkimu suma:\n{prekiu_pardavimai_metams}')

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Year', y='Total Purchase Amount', hue='Product Category', palette='husl')
plt.title('Pardavimai per metus')
plt.xlabel('Metai')
plt.ylabel('Pardavimų suma')
plt.legend()
# plt.show()


##pardavimai pagal kategorijas per menesi
df['Month'] = df['Purchase Date'].dt.month

prekiu_pardavimai_menesiui = df.groupby(['Month', 'Product Category'])['Total Purchase Amount'].sum()
# print(f'Prekiu pardavimai per menesi pagal kategorijas ir bendra pirkimu suma:\n{prekiu_pardavimai_menesiui}')

plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='Month', y='Total Purchase Amount', hue='Product Category', palette='husl')
plt.title('Pardavimai per mėnesį')
plt.xlabel('Mėnesis')
plt.ylabel('Pardavimų suma')
plt.legend()
# plt.show()


##labiausiai perkamos prekiu kategorijos
kategoriju_pirkimu_suma = df.groupby('Product Category')['Total Purchase Amount'].sum()


##kategorijos pagal pirkimu suma mazejimo tvarka
perkamiausios_prekiu_kategorijos = kategoriju_pirkimu_suma.sort_values(ascending=False)
# print(f'Prekiu kategoriju perkamumas: \n{perkamiausios_prekiu_kategorijos}')


##1eB vienas eksabajtas. Tai yra duomenų saugojimo matavimo vienetas, kuris lygus 10^18 baitų. Todėl skaičiai per kablelį..
plt.figure(figsize=(12, 6))
sns.barplot(x=perkamiausios_prekiu_kategorijos.values, y=perkamiausios_prekiu_kategorijos.index, palette='husl')
plt.title('Labiausiai perkamų prekių kategorijos')
plt.xlabel('Bendra pirkimų suma')
plt.ylabel('Kategorija')
plt.show()


#_______________________________________________________

#---Analizuojame populiariausius atsiskaitymo budus---

atsiskaitymo_budas = (df['Payment Method'].value_counts()
                                       .plot(kind='pie', y='popurliariausias_atsiskaitymo_budas', autopct='%1.0f%%',
                                             colors=['pink', 'skyblue', 'gold', 'orchid']))
plt.title('Atsiskaitymo būdų analizė')
plt.show()

#Atsiskaitymo budu suskaiciavimas nuo didziausio iki maziausio
populiariausias_atsiskaitymo_budas = df['Payment Method'].value_counts()
print(populiariausias_atsiskaitymo_budas)

#---inventorizacijos kiekio plaiakymas pagal kategorija---
prekiu_kiekis_pagal_kategorija = df.groupby('Product Category')['Quantity'].sum()
print(prekiu_kiekis_pagal_kategorija)

#---kurie klientai gryzta dazniausiai---

top_10_klientai_gryzta_dazniausiai = df['Customer Name'].value_counts().head(10)
print(top_10_klientai_gryzta_dazniausiai)

#---Kiek klientu renkasi prenumerata---

klientu_prenumerata = (df['Churn'].value_counts().plot(kind='pie', y='klientu_prenumerata', autopct='%1.0f%%',
                                             colors=['pink', 'skyblue', 'gold', 'orchid']))
plt.title('Klientų prenumeratų analizė')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1), labels=["0-Atsisakė", "1-Sutiko"])
plt.show()

klientu_prenumerata = df['Churn'].value_counts()
print(klientu_prenumerata)

