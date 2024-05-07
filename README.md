# E-Parduotuvės klientų elgsenos analizė ir prognozavimas

### Projekto autorės:
- Samanta Čečkauskaitė
- Rasa Dzenkauskaitė
## Įvadas:
Analizuoti e-parduotuvės klientų pirkimo elgseną ir prognozuoti būsimus pirkimų kiekius, taip padedant verslui geriau suprasti klientų poreikius.


### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

### Technologijos:
Python ✦ TensorFlow/Keras ✦ Plotly ✦ Seaborn ✦ Scikit-Learn ✦ Pandas ✦ Sklearn ✦ Numpy ✦ Matplotlib ✦

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

## Darbo etapai:

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## 1.Duomenų valymas ir paruošimas naudojimui:

```javascript

#--Nuskaitome duomenis
df = pd.read_csv('ecommerce_customer_data_custom_ratios.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
# print(df.describe())


#--Tikriname eilučių skaičių
# print(len(df))


#--Tikriname NaN reikšmes
# print(df.isnull().sum())


#--Sutvarkome "Returns" skiltį, pašaliname NaN reikšmes
df['Returns'] = df['Returns'].fillna(0).astype(int)

# --Pašaliname 'customer age' stulpelį, nes jis identiškas 'age' stulpeliui--
df = df.drop('Customer Age', axis=1)

#--Konvertuojame pirkimo datą į dataframe
df['Purchase Date'] = pd.to_datetime(df['Purchase Date'])
df['Purchase Year'] = df['Purchase Date'].dt.year
df['Purchase Month'] = df['Purchase Date'].dt.month_name()
```

## 2. Duomenų analizė pagal klientus:

Analaziuojame kurios lyties klientai perka daugiausiai ir matome, kad abi lytis išleidžia panašią sumą pinigų.

| Nr |	Lytis |	Suma |
|-------|------|--------------|
|1  |Moteris	|342462421
|2	|Vyras	|338880262

Analizuojame klientų kiekį pagal amžiaus grupes ir matome, kad didžiausią mūsų klientų kiekį sudaro nuo 25 m. iki 50 metų grupės, o mažiausią kiekį sudaro klientai iki 25 metų.
Tai mums leidžią žinoti, kurio amžiaus žmonės apsiperka daugiausiai.

<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/047d2e07-10ef-4f3b-9cdb-0bb2fcf66d49" 
     width="480" 
     height="360" />

  |Klientų skirstymas|	Išleistos sumos vidurkis |
|-------|--------------|
|Klientai jaunesni nei 25 m.| 33671
|Klientai nuo 25 m. iki 50 m.|112988
|Klientai nuo 50 m.|94081

Analizuojame klientų pasirinkimą į prenumeratą/naujienlaiškį ir matome, kad tik 20% visų klientų renkasi gauti naujienlaiškį. Iš to galime spręsti, kad E parduotuvei paslauga nenaudinga.

<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/55907cdc-5cb3-43bf-81d5-2b47ce6172df"
     width="480" 
     height="360" />

| Naujienlaiškis |	Statistika |
|-------|------|
| Nesirenka |200126
|Renkasi  |49874

Klientų atsiskaitymo būdų analizė. Matome, kad klientai labiau linkę atsiskaityti Kreditine kortele 40% ir PayPal 30%. Siūlome reklamuoti, kad E parduotuvėje galima atsiskaityti šitais būdais.

<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/bb35c427-e60a-4e71-bc3b-cbd65127a23a"
     width="480" 
     height="360" />

| Mokėjimo metodas |	Statistika |
|-------|------|
|Credit Card   |100486
|PayPal          |74837
|Cash            |49894
|Crypto          |24783


## Duomenų analizė pagal pirkimus:

Labiausiai perkamos prekių kategorijos per visą laikotarpį. Matome, kad daugiausiai perka knygas ir rūbus, tad siūlytume juos reklamuoti daugiausiai ir turėti daugiau inventoriaus šiom kategorijom.

<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/b3d3172a-a558-4cfc-8d12-1e9aa4392d84"
     width="500" 
     height="360" />
     
| Nr. | Prekių kategorijos | Statistika 
|----|-------|------|
| 1. |Books          |223876
| 2. |Clothing       |225322
| 3. |Electronics    |150828
| 4. |Home           |149698

Apžiūrime pardavimus per mėnesį ir metus.


<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/8f327f1d-096d-4fb7-8ee9-f67ac3eb13d0"
     width="500" 
     height="360" />
<img src="https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/c3319852-2cc1-4584-8c2e-185d036cabaf"
     width="500" 
     height="360" />

### - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
## 3. Klientų segmentacija su KMeans

Skirtingi klasteriai pagal spalvų juostą dešinėje atspindi kaip klientai gali būti grupuojami pagal išlaidų elgseną ir dažnumą, kas gali būti vertinga rinkos segmentavimo analizei. Iš to matome, kad yra trys klientų grupės atsižvelgiant į išleistą bendrą pinigų kiekį ir išleistą pinigų kiekį per pirkimą. Geltona grupė - išleidusi bendrai mažai pinigų ir mažai išleidžią per pirkimą. Violetinė grupė - išleidusi bendrai daug pinigų ir per pirkimą išleidžia šiek tiek daugiau už geltoną grupę žmonių. Mėlyna grupė - išleidusi bendrai mažai pinigų, bet per pirkimą išleidžia daug. Bendrai atsižvelgiant, E parduotuvei reikėtų sutelkti dėmesį į violetinės grupės žmones, nes jie ne vien yra sugrįžtantys pirkėjai, bet ir išleidžiantys nemažą sumą per pirkimą, kas padaro juos didžiausio pelno šaltiniu.

| Vertinimas. | Klasteriai | Gaunamas rezultatas 
|----|-------|------|
|K-Means silhouette score: |k= 3| 0.34 
|K-Means silhouette score: |k= 6| 0.29
|K-Means silhouette score: |k= 9| 0.26
|Best silhouette score for k = 3

![image](https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/97bba6ff-48e0-446b-b022-152cf63fef49)


## 4. Modelio kūrimas

<b> LSTM modelis "Faktinės ir prognozuotos pirkimų reikšmės" </b>

| Metrika | Rezultatas |
|----|-------|
|MSE: |2.001457929611206
|R2: |-0.00015582927564650184
|MAE: |1.2036666989294689
![newplot(4)](https://github.com/Samantjna/E-Store-Customer-Behavior-Analysis-and-Forecasting/assets/163418549/a33a94ba-5a4d-4e27-8972-24b4730bda9c)








