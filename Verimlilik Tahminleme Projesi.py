Python 3.13.3 (tags/v3.13.3:6280bb5, Apr  8 2025, 14:47:33) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
# Pandas kütüphanesi
import pandas as pd
# Dataset dosyasını yükleme
df = pd.read_csv(
    r"C:\Users\LENOVO\OneDrive\Masaüstü\dataset.csv",
    encoding='ISO-8859-1',
    low_memory=False
)
# İlk 5 satıra bakalım
df.head()
     sub_ID  sub_fname  ... recorded_note_from_sup  record_conf_matrix_h
0  98000001    Rebecca  ...                    NaN                   NaN
1  98000001    Rebecca  ...                    NaN                   NaN
2  98000002       Joan  ...                    NaN                   NaN
3  98000002       Joan  ...                    NaN                   NaN
4  98000003  Elizabeth  ...                    NaN                   NaN

[5 rows x 42 columns]
# Veri tipi kontrolü
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 411948 entries, 0 to 411947
Data columns (total 42 columns):
 #   Column                   Non-Null Count   Dtype  
---  ------                   --------------   -----  
 0   sub_ID                   411948 non-null  int64  
 1   sub_fname                411948 non-null  object 
 2   sub_lname                411948 non-null  object 
 3   sub_age                  411948 non-null  int64  
 4   sub_sex                  411948 non-null  object 
 5   sub_shift                411948 non-null  object 
 6   sub_team                 411948 non-null  object 
 7   sub_role                 411948 non-null  object 
 8   sub_coll_IDs             411136 non-null  object 
 9   sub_colls_same_sex_prtn  410957 non-null  float64
 10  sub_health_h             411948 non-null  float64
 11  sub_commitment_h         411948 non-null  float64
 12  sub_perceptiveness_h     411948 non-null  float64
 13  sub_dexterity_h          411948 non-null  float64
 14  sub_sociality_h          411948 non-null  float64
 15  sub_goodness_h           411948 non-null  float64
 16  sub_strength_h           411948 non-null  float64
 17  sub_openmindedness_h     411948 non-null  float64
 18  sub_workstyle_h          411948 non-null  object 
 19  sup_ID                   411136 non-null  float64
 20  sup_fname                411136 non-null  object 
 21  sup_lname                411136 non-null  object 
 22  sup_age                  411136 non-null  float64
 23  sup_sub_age_diff         411136 non-null  float64
 24  sup_sex                  411136 non-null  object 
 25  sup_role                 411136 non-null  object 
 26  sup_commitment_h         411136 non-null  float64
 27  sup_perceptiveness_h     411136 non-null  float64
 28  sup_goodness_h           411136 non-null  float64
 29  event_date               411948 non-null  object 
 30  event_week_in_series     411948 non-null  int64  
 31  event_day_in_series      411948 non-null  int64  
 32  event_weekday_num        411948 non-null  int64  
 33  event_weekday_name       411948 non-null  object 
 34  behav_comptype_h         411846 non-null  object 
 35  behav_cause_h            77 non-null      object 
 36  actual_efficacy_h        191657 non-null  float64
 37  record_comptype          408010 non-null  object 
 38  record_cause             102 non-null     object 
 39  recorded_efficacy        191272 non-null  float64
 40  recorded_note_from_sup   18589 non-null   object 
 41  record_conf_matrix_h     21715 non-null   object 
dtypes: float64(17), int64(5), object(20)
memory usage: 132.0+ MB
# Hedef değişkeni içermeyen satırları sil
df_model = df[df['actual_efficacy_h'].notnull()].copy()
# Aşırı boş olan sütunları sil
df_model.drop(columns=[
    'behav_cause_h',
    'record_cause',
    'recorded_note_from_sup',
    'record_conf_matrix_h'
], inplace=True)
df_model.shape
(191657, 38)
# Kategorik verileri sayı formuna çevirme
df_model.select_dtypes(include='object').columns.tolist()
['sub_fname', 'sub_lname', 'sub_sex', 'sub_shift', 'sub_team', 'sub_role', 'sub_coll_IDs', 'sub_workstyle_h', 'sup_fname', 'sup_lname', 'sup_sex', 'sup_role', 'event_date', 'event_weekday_name', 'behav_comptype_h', 'record_comptype']
# Model için gereksiz sütunları temizleme
categorical_cols = [
    'sub_sex', 'sub_shift', 'sub_team', 'sub_role', 'sub_workstyle_h', 'sup_sex', 'sup_role', 'event_weekday_name', 'behav_comptype_h', 'record_comptype'
]
df_encoded = pd.get_dummies(df_model, columns=categorical_cols, drop_first=True)
df_encoded.shape
(191657, 71)
#Eksik değer kontrolü
missing_counts = df_encoded.isnull().sum()
missing_ratio = (missing_counts / len(df_encoded)) * 100
missing_df = pd.DataFrame({'missing_count': missing_counts, 'missing_ratio_%':missing_ratio})
print(missing_df[missing_df['missing_count'] > 0].sort_values(by='missing_ratio_%', ascending=False))
                         missing_count  missing_ratio_%
sub_coll_IDs                       385          0.20088
sub_colls_same_sex_prtn            385          0.20088
sup_ID                             385          0.20088
sup_fname                          385          0.20088
sup_lname                          385          0.20088
sup_age                            385          0.20088
sup_sub_age_diff                   385          0.20088
sup_commitment_h                   385          0.20088
sup_perceptiveness_h               385          0.20088
sup_goodness_h                     385          0.20088
recorded_efficacy                  385          0.20088
print(df_encoded['actual_efficacy_h'].describe())
count    191657.000000
mean          0.669063
std           0.396246
min           0.000000
25%           0.410000
50%           0.615000
75%           0.874000
max           3.763000
Name: actual_efficacy_h, dtype: float64
print(f"actual_efficacy_h null sayısı: {df_encoded['actual_efficacy_h'].isnull().sum()}")
actual_efficacy_h null sayısı: 0
# Eksik değerleri doldurma
categorical_cols_to_fill = ['sub_coll_IDs', 'sup_fname', 'sup_lname']
for col in categorical_cols_to_fill:
    mode_val = df_encoded[col].mode()[0]
    df_encoded[col] = df_encoded[col].fillna(mode_val)

    
numerical_cols_to_fill = [
    'sub_colls_same_sex_prtn', 'sup_ID', 'sup_age', 'sup_sub_age_diff', 'sup_commitment_h', 'sup_perceptiveness_h', 'recorded_efficacy'
]
for col in numerical_cols_to_fill:
    median_val = df_encoded[col].median()
    df_encoded[col] = df_encoded[col].fillna(median_val)

    
print(df_encoded.isnull().sum().sort_values(ascending=False).head(10))
sup_goodness_h             385
sub_fname                    0
sub_ID                       0
sub_age                      0
sub_coll_IDs                 0
sub_colls_same_sex_prtn      0
sub_lname                    0
sub_health_h                 0
sub_commitment_h             0
sub_dexterity_h              0
dtype: int64
median_val =df_encoded['sup_goodness_h'].median()
df_encoded['sup_goodness_h'] = df_encoded['sup_goodness_h'].fillna(median_val)
print(df_encoded.isnull().sum().sort_values(ascending=False).head(10))
sub_ID                     0
sub_fname                  0
sub_lname                  0
sub_age                    0
sub_coll_IDs               0
sub_colls_same_sex_prtn    0
sub_health_h               0
sub_commitment_h           0
sub_perceptiveness_h       0
sub_dexterity_h            0
dtype: int64
non_numeric_cols = x_train.select_dtypes(include=['object']).columns
Traceback (most recent call last):
  File "<pyshell#35>", line 1, in <module>
    non_numeric_cols = x_train.select_dtypes(include=['object']).columns
NameError: name 'x_train' is not defined
# Modelleme öncesi son kontroller
from sklearn.model_selection import train_test_split
y = df_encoded['actual_efficacy_h']
x = df_encoded.drop(columns=['actual_efficacy_h'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print(f"Train shape: {x_train.shape}, Test shape: {x_test.shape}")
Train shape: (153325, 70), Test shape: (38332, 70)
non_numeric_cols = x_train.select_dtypes(include=['object']).columns
print(non_numeric_cols)
Index(['sub_fname', 'sub_lname', 'sub_coll_IDs', 'sup_fname', 'sup_lname',
       'event_date'],
      dtype='object')
for col in ['sub_fname', 'sub_lname', 'sub_coll_IDs', 'sup_fname', 'sup_lname', 'sub_coll_IDs']:
    print (f"{col}:(df_encoded[col].nunique()}kategori")
    
SyntaxError: f-string: single '}' is not allowed
for col in ['sub_fname', 'sub_lname', 'sub_coll_IDs', 'sup_fname', 'sup_lname', 'sub_coll_IDs']:
    print (f"{col}:{(df_encoded[col].nunique()}kategori")
    
SyntaxError: closing parenthesis '}' does not match opening parenthesis '('
for col in ['sub_fname', 'sub_lname', 'sub_coll_IDs', 'sup_fname', 'sup_lname', 'sub_coll_IDs']:
    print(f"{col}:{df_encoded[col].nunique()}kategori")

sub_fname:99kategori
sub_lname:100kategori
sub_coll_IDs:3786kategori
sup_fname:26kategori
sup_lname:31kategori
sub_coll_IDs:3786kategori
for col in ['sub_fname', 'sub_lname', 'sub_coll_IDs', 'sup_fname', 'sup_lname', 'event_date']:
    print(f"{col}:{df_encoded[col].nunique()}kategori")

    
sub_fname:99kategori
sub_lname:100kategori
sub_coll_IDs:3786kategori
sup_fname:26kategori
sup_lname:31kategori
event_date:435kategori
df_encoded = pd.get_dummies(df_encoded, columns=['sub_fname', 'sub_lname', 'sup_fname', 'sup_lname', 'event_date'], drop_first=True)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_encoded['sub_coll_IDs'] = le.fit_transform(df_encoded['sub_coll_IDs'])
for col in x.select_dtypes(include='bool').columns:
    x[col] = x[col].astype(int)

    
print(x.dtypes.unique())
[dtype('int64') dtype('O') dtype('float64')]
non_numeric_cols = x_train.select_dtypes(include=['object']).columns
print("Sayısal olmayan sütunlar:", list(non_numeric_cols))
Sayısal olmayan sütunlar: ['sub_fname', 'sub_lname', 'sub_coll_IDs', 'sup_fname', 'sup_lname', 'event_date']
from  sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in non_numeric_cols:
    try:
        x[col] = le.fit_transform(x[col])
    expect:
        
SyntaxError: expected 'except' or 'finally' block
from  sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in non_numeric_cols:
    try:
        x[col] = le.fit_transform(x[col])
    except:
        
SyntaxError: multiple statements found while compiling a single statement
from  sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in non_numeric_cols:
    try:
        x[col] = le.fit_transform(x[col])
    except:
        print(f"{col} sütununda dönüşüm başarısız!")

print(x.dtypes.unique())
[dtype('int64') dtype('float64')]
y = df_encoded['actual_efficacy_h']
x = df_encoded.drop(columns=['actual_efficacy_h'])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
lr_model.fit(x_train, y_train)
LinearRegression()
from sklearn.metrics import mean_squared_error, r2_score
y_pred = lr_model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
Mean Squared Error: 0.01
print (f"R^2 Score: {r2:.4f}")
R^2 Score: 0.9220
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
<Figure size 800x600 with 0 Axes>
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
<matplotlib.collections.PathCollection object at 0x0000018B4F5C96A0>
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
[<matplotlib.lines.Line2D object at 0x0000018B4F5A56D0>]
plt,slabel('Gerçek Değerler')
Traceback (most recent call last):
  File "<pyshell#94>", line 1, in <module>
    plt,slabel('Gerçek Değerler')
NameError: name 'slabel' is not defined
plt.xlabel('Gerçek Değerler')
Text(0.5, 0, 'Gerçek Değerler')
plt.ylabel('Tahmin Edilen Değerler)
           
SyntaxError: unterminated string literal (detected at line 1)
plt.ylabel('Tahmin Edilen Değerler')
           
Text(0, 0.5, 'Tahmin Edilen Değerler')
plt.title('Gerçek Değerler - Tahmin Edilen Değerler)
          
SyntaxError: unterminated string literal (detected at line 1)
plt.title('Gerçek Değerler - Tahmin Edilen Değerler')
          
Text(0.5, 1.0, 'Gerçek Değerler - Tahmin Edilen Değerler')
plt.grid(True)
          
plt.show()
          
import matplotlib.pyplot as plt
import pandas as pd
coefficients = pd.Series(lr_model.coef_, index=x_train.columns)
top_positive = coefficients.sort_values(ascending=False).head(10)
top_negative = coefficients.sort_values().head(10)
plt.figure(figsize=12,6))
SyntaxError: unmatched ')'
plt.figure(figsize=(12,6))
<Figure size 1200x600 with 0 Axes>
plt.subplot(1, 2, 1)
<Axes: >
top_positive.plot(kind='barh', color='green')
<Axes: >
plt.title('Pozitif Etkili Özellikler')
Text(0.5, 1.0, 'Pozitif Etkili Özellikler')
plt.xlabel('Koefisiyent Değeri')
Text(0.5, 0, 'Koefisiyent Değeri')
plt.grid(True)
plt.subplot(1, 2, 2)
<Axes: >
top_negative.plot(kind='barh', color='red')
<Axes: >
plt.xlabel('Koefisiyent Değeri')
Text(0.5, 0, 'Koefisiyent Değeri')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.subplot(1, 2, 2)
<Axes: >
top_negative.plot(kind='barh', color='red')
<Axes: >
plt.title('Negatif Etkili Özellikler')
Text(0.5, 1.0, 'Negatif Etkili Özellikler')
plt.xlabel('Koefisiyent Değeri')
Text(0.5, 0, 'Koefisiyent Değeri')
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure(figsize=(12,6))
<Figure size 1200x600 with 0 Axes>
plt.subplot(1, 2, 1)
<Axes: >
top_positive.plot(kind='barh', color='green')
<Axes: >
plt.title('Pozitif Etkili Özellikler')
Text(0.5, 1.0, 'Pozitif Etkili Özellikler')
plt.xlabel('Koefisiyent Değeri')
Text(0.5, 0, 'Koefisiyent Değeri')
plt.grid(True)
plt.subplot(1, 2, 2)
<Axes: >
top_negative.plot(kind='barh', color='red')
<Axes: >
plt.title('Negatif Etkili Özellikler')
Text(0.5, 1.0, 'Negatif Etkili Özellikler')
plt.xlabel('Koefisiyent Değeri')
Text(0.5, 0, 'Koefisiyent Değeri')
plt.grid(True)
plt.tight_layout()
plt.show()
import pandas as pd
coefficients = pd.Series(lr_model.coef_, index=x_train.columns)
top_positive = coefficients.sort_values(ascending=False).head(10)
top_negative = coefficients.sort_values().head(10)
print("✅ Pozitif Etkili Özellikler:\n")
✅ Pozitif Etkili Özellikler:

print(top_positive)
recorded_efficacy        0.903723
event_date_12/25/2021    0.236784
event_date_1/9/2021      0.188473
event_date_11/13/2021    0.173096
event_date_10/23/2021    0.146109
event_date_12/11/2021    0.143680
event_date_5/29/2021     0.104593
event_date_6/19/2021     0.102942
event_date_3/5/2022      0.094777
event_date_1/16/2021     0.089766
dtype: float64
print("\n❌ Negatif Etkili Özellikler:\n")

❌ Negatif Etkili Özellikler:

print(top_negative)
event_date_2/6/2021     -0.328181
event_date_1/22/2022    -0.199223
event_date_7/31/2021    -0.189642
event_date_9/4/2021     -0.114413
event_date_3/26/2022    -0.100276
event_date_11/27/2021   -0.095861
event_date_6/12/2021    -0.091208
event_date_4/23/2022    -0.062368
event_date_2/12/2022    -0.062230
event_date_5/8/2021     -0.056803
dtype: float64
for i, (feature, coef) in enumerate(top_positive.items(), 1):
    print(f"{i}. {feature}: +{coef:.4f} ✅")

    
1. recorded_efficacy: +0.9037 ✅
2. event_date_12/25/2021: +0.2368 ✅
3. event_date_1/9/2021: +0.1885 ✅
4. event_date_11/13/2021: +0.1731 ✅
5. event_date_10/23/2021: +0.1461 ✅
6. event_date_12/11/2021: +0.1437 ✅
7. event_date_5/29/2021: +0.1046 ✅
8. event_date_6/19/2021: +0.1029 ✅
9. event_date_3/5/2022: +0.0948 ✅
10. event_date_1/16/2021: +0.0898 ✅
for i, (feature, coef) in enumerate(top_negative.items(), 1):
    print(f"{i}. {feature}: +{coef:.4f} ❌")

    
1. event_date_2/6/2021: +-0.3282 ❌
2. event_date_1/22/2022: +-0.1992 ❌
3. event_date_7/31/2021: +-0.1896 ❌
4. event_date_9/4/2021: +-0.1144 ❌
5. event_date_3/26/2022: +-0.1003 ❌
6. event_date_11/27/2021: +-0.0959 ❌
7. event_date_6/12/2021: +-0.0912 ❌
8. event_date_4/23/2022: +-0.0624 ❌
9. event_date_2/12/2022: +-0.0622 ❌
10. event_date_5/8/2021: +-0.0568 ❌
for i, (feature, coef) in enumerate(top_negative.items(), 1):
    print(f"{i}. {feature}: {coef:.4f} ❌")

           
1. event_date_2/6/2021: -0.3282 ❌
2. event_date_1/22/2022: -0.1992 ❌
3. event_date_7/31/2021: -0.1896 ❌
4. event_date_9/4/2021: -0.1144 ❌
5. event_date_3/26/2022: -0.1003 ❌
6. event_date_11/27/2021: -0.0959 ❌
7. event_date_6/12/2021: -0.0912 ❌
8. event_date_4/23/2022: -0.0624 ❌
9. event_date_2/12/2022: -0.0622 ❌
10. event_date_5/8/2021: -0.0568 ❌
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
top_positive.sort_values().plot(kind='barh', color='green')
<Axes: >
plt.title('✅ Pozitif Etkili Özellikler')
Text(0.5, 1.0, '✅ Pozitif Etkili Özellikler')
plt.xlabel(Katsayı Değeri')
           
SyntaxError: unterminated string literal (detected at line 1)
plt.xlabel('Katsayı Değeri')
           
Text(0.5, 0, 'Katsayı Değeri')
plt.grid(True)
           
plt.tight_layout()
           

Warning (from warnings module):
  File "<pyshell#163>", line 1
UserWarning: Glyph 9989 (\N{WHITE HEAVY CHECK MARK}) missing from font(s) DejaVu Sans.
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
top_positive.sort_values().plot(kind='barh', color='green')
<Axes: >
plt.title('✔ Pozitif Etkili Özellikler')
Text(0.5, 1.0, '✔ Pozitif Etkili Özellikler')
plt.xlabel('Katsayı Değeri')
Text(0.5, 0, 'Katsayı Değeri')
plt.grid(True)
plt.tight_layout()
plt.show()

Warning (from warnings module):
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\tkinter\__init__.py", line 862
    func(*args)
UserWarning: Glyph 9989 (\N{WHITE HEAVY CHECK MARK}) missing from font(s) DejaVu Sans.
plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
top_negative.sort_values().plot(kind='barh', color='red')
<Axes: >
plt.title('❌ Negatif Etkili Özellikler')
Text(0.5, 1.0, '❌ Negatif Etkili Özellikler')
plt.xlabel('Katsayı Değeri')
Text(0.5, 0, 'Katsayı Değeri')
plt.grid(True)
plt.tight_layout()

Warning (from warnings module):
  File "<pyshell#177>", line 1
UserWarning: Glyph 10060 (\N{CROSS MARK}) missing from font(s) DejaVu Sans.
plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
top_negative.sort_values().plot(kind='barh', color='red')
<Axes: >
plt.title('✗ Negatif Etkili Özellikler')
Text(0.5, 1.0, '✗ Negatif Etkili Özellikler')
plt.xlabel('Katsayı Değeri')
Text(0.5, 0, 'Katsayı Değeri')
plt.grid(True)
plt.tight_layout()
plt.show()

Warning (from warnings module):
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\tkinter\__init__.py", line 862
    func(*args)
UserWarning: Glyph 10060 (\N{CROSS MARK}) missing from font(s) DejaVu Sans.
event_date_cols = [col for col in coefficients.index if col.startswith('event_date_')]
event_date_coeffs = coefficients[event_date_cols]
event_date_series = event_date_coeffs.copy()
event_date_series.index = [col.replace('event_date_', '') for col in event_date_series.index]
event_date_series.index = pd.to_datetime(event_date_series.index)
import matplotlib.pyplot as plt
plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
event_date_series.sort_index().plot(marker='o')
<Axes: >
plt.axhline(0, color='gray', linestyle='--')
<matplotlib.lines.Line2D object at 0x0000018B5D371D10>
plt.title("Tarihlere Göre Model Katsayıları (Etki Analizi)")
Text(0.5, 1.0, 'Tarihlere Göre Model Katsayıları (Etki Analizi)')
plt.xlabel("Tarih")
Text(0.5, 0, 'Tarih')
plt.ylabel("Katsayı")
Text(0, 0.5, 'Katsayı')
plt.grid(True)
plt.tight_layout()
plt.show()
import pandas as pd
df = pd.read_csv('data.csv')
Traceback (most recent call last):
  File "<pyshell#201>", line 1, in <module>
    df = pd.read_csv('data.csv')
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1880, in _make_engine
    self.handles = get_handle(
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'
df = pd.read_csv('dataset.csv')
Traceback (most recent call last):
  File "<pyshell#202>", line 1, in <module>
    df = pd.read_csv('dataset.csv')
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1026, in read_csv
    return _read(filepath_or_buffer, kwds)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 620, in _read
    parser = TextFileReader(filepath_or_buffer, **kwds)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1620, in __init__
    self._engine = self._make_engine(f, self.engine)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\parsers\readers.py", line 1880, in _make_engine
    self.handles = get_handle(
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\common.py", line 873, in get_handle
    handle = open(
FileNotFoundError: [Errno 2] No such file or directory: 'dataset.csv'
df = pd.read_csv(
    r"C:\Users\LENOVO\OneDrive\Masaüstü\dataset.csv",
    encoding='ISO-8859-1',
    low_memory=False
)
df['event_date'] = pd.to_datetime(df['event_date'], format='%m/%d/%Y')
df['num_collaborators'] = df['sub_coll_IDs'].apply(lambda x: len(eval(x)) if pd.notnull(x) else 0)
drop_cols = [
    'sub_ID', 'sub_fname', 'sub_lname',
    'sup_ID', 'sup_fname', 'sup_lname',
    'recorded_note_from_sup', 'record_conf_matrix_h'
]
df = df.drop(columns=drop_cols)
categorical_cols = ['sub_sex', 'sub_shift', 'sub_team', 'sub_role', 'record_comptype', 'record_cause', 'event_weekday_name', 'behav_comptype_h', 'behav_cause_h']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

missing = df.isna().sum()
print("Eksik değerler:\n", missing[missing > 0])
SyntaxError: multiple statements found while compiling a single statement
missing = df.isna().sum()
print("Eksik değerler:\n", missing[missing > 0])
Eksik değerler:
 sub_coll_IDs                  812
sub_colls_same_sex_prtn       991
sup_age                       812
sup_sub_age_diff              812
sup_sex                       812
sup_role                      812
sup_commitment_h              812
sup_perceptiveness_h          812
sup_goodness_h                812
behav_comptype_h              102
behav_cause_h              411871
actual_efficacy_h          220291
record_comptype              3938
record_cause               411846
recorded_efficacy          220676
dtype: int64
df.to_excel('cleaned_data.xlsx', index=False)
Traceback (most recent call last):
  File "<pyshell#215>", line 1, in <module>
    df.to_excel('cleaned_data.xlsx', index=False)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\util\_decorators.py", line 333, in wrapper
    return func(*args, **kwargs)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\generic.py", line 2417, in to_excel
    formatter.write(
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\formats\excel.py", line 943, in write
    writer = ExcelWriter(
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\io\excel\_openpyxl.py", line 57, in __init__
    from openpyxl.workbook import Workbook
ModuleNotFoundError: No module named 'openpyxl'
df.to_excel('cleaned_data.xlsx', index=False)
  
print("Temizlenmiş dosya 'cleaned_data.xlsx' olarak kaydedildi.")
  
Temizlenmiş dosya 'cleaned_data.xlsx' olarak kaydedildi.
import pandas as pd
df = pd.read_excel('cleaned_data.xlsx')
print(df.head())
   sub_age sub_sex sub_shift  ... record_cause recorded_efficacy num_collaborators
0       40       F   Shift 1  ...          NaN               NaN                 7
1       40       F   Shift 1  ...          NaN               1.2                 7
2       61       F   Shift 1  ...          NaN               NaN                 7
3       61       F   Shift 1  ...          NaN               0.8                 7
4       20       F   Shift 1  ...          NaN               NaN                 7

[5 rows x 35 columns]
import pandas as pd
df['event_date'] = pd.to_datetime(df['event_date'], dayfirst=True)
mart_2021 = df[(df['event_date'].dt.year == 2021) & (df['event_date'].dt.month == 3)]
ocak_2022 = df[(df['event_date'].dt.year == 2022) & (df['event_date'].dt.month == 1)]
print(f"2021 Mart kayıt sayısı: {len(mart_2021)}")
2021 Mart kayıt sayısı: 24351
print(f"2022 Ocak kayıt sayısı: {len(ocak_2022)}")
2022 Ocak kayıt sayısı: 22174
mart_2021.to_excel('mart_2021.xlsx', index=False)
ocak_2022.to_excel('ocak_2022.xlsx', index=False)
import os
print(os.getcwd())
C:\Users\LENOVO\AppData\Local\Programs\Python\Python313
import pandas as pd
df['date'] = pd.to_datetime(df['date'])
Traceback (most recent call last):
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\indexes\base.py", line 3805, in get_loc
    return self._engine.get_loc(casted_key)
  File "index.pyx", line 167, in pandas._libs.index.IndexEngine.get_loc
  File "index.pyx", line 196, in pandas._libs.index.IndexEngine.get_loc
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 7081, in pandas._libs.hashtable.PyObjectHashTable.get_item
  File "pandas\\_libs\\hashtable_class_helper.pxi", line 7089, in pandas._libs.hashtable.PyObjectHashTable.get_item
KeyError: 'date'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "<pyshell#232>", line 1, in <module>
    df['date'] = pd.to_datetime(df['date'])
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\frame.py", line 4102, in __getitem__
    indexer = self.columns.get_loc(key)
  File "C:\Users\LENOVO\AppData\Local\Programs\Python\Python313\Lib\site-packages\pandas\core\indexes\base.py", line 3812, in get_loc
    raise KeyError(key) from err
KeyError: 'date'
print(df.columns)
Index(['sub_age', 'sub_sex', 'sub_shift', 'sub_team', 'sub_role',
       'sub_coll_IDs', 'sub_colls_same_sex_prtn', 'sub_health_h',
       'sub_commitment_h', 'sub_perceptiveness_h', 'sub_dexterity_h',
       'sub_sociality_h', 'sub_goodness_h', 'sub_strength_h',
       'sub_openmindedness_h', 'sub_workstyle_h', 'sup_age',
       'sup_sub_age_diff', 'sup_sex', 'sup_role', 'sup_commitment_h',
       'sup_perceptiveness_h', 'sup_goodness_h', 'event_date',
       'event_week_in_series', 'event_day_in_series', 'event_weekday_num',
       'event_weekday_name', 'behav_comptype_h', 'behav_cause_h',
       'actual_efficacy_h', 'record_comptype', 'record_cause',
       'recorded_efficacy', 'num_collaborators'],
      dtype='object')
df['event_date'] = pd.to_datetime(df['event_date'])
df['month'] = df['event_date'].dt.to_period('M')
monthly_efficiency = df.groupby('month')['recorded_efficacy'].mean().reset_index()
print(monthly_efficiency)
      month  recorded_efficacy
0   2021-01           0.743536
1   2021-02           0.698866
2   2021-03           0.682297
3   2021-04           0.670003
4   2021-05           0.621770
5   2021-06           0.599648
6   2021-07           0.608776
7   2021-08           0.629151
8   2021-09           0.664306
9   2021-10           0.689751
10  2021-11           0.698619
11  2021-12           0.756190
12  2022-01           0.730269
13  2022-02           0.693948
14  2022-03           0.679899
15  2022-04           0.660357
16  2022-05           0.613427
17  2022-06           0.604179
max_value = event_date_series.max()
max_date = event_date_series.idxmax()
min_value = event_date_series.min()
min_date = event_date_series.idxmin()
print(f"Pozitif tepe noktası: {max_value} tarihinde {max_date}")
Pozitif tepe noktası: 0.23678404930414695 tarihinde 2021-12-25 00:00:00
print(f"Negatif tepe noktası: {min_value} tarihinde {min_date}")
Negatif tepe noktası: -0.328181050544852 tarihinde 2021-02-06 00:00:00
df['event_date'] = pd.to_datetime(df['event_date'])
start_neg = pd.to_datetime('2021-02-06') - pd.Timedelta(days=7)
end_neg = pd.to_datetime('2021-02-06') + pd.Timedelta(days=7)
neg_period = df[(df['event_date'] >= start_neg) & (df['event_date'] <= end_neg)]
start_pos = pd.to_datetime('2021-12-25') - pd.Timedelta(days=7)
end_pos = pd.to_datetime('2021-12-25') + pd.Timedelta(days=7)
pos_period = df[(df['event_date'] >= start_pos) & (df['event_date'] <= end_pos)]
print("Negatif Dönem Vardiya Dağılımı:")
Negatif Dönem Vardiya Dağılımı:
print(neg_period['sub_shift'].value_counts())
sub_shift
Shift 2       3508
Shift 1       3498
Shift 3       3494
unassigned      21
Name: count, dtype: int64
print("\nNegatif Dönem Takım Dağılımı:")

Negatif Dönem Takım Dağılımı:
print(neg_period['sub_team'].value_counts())
sub_team
Team 14       447
Team 6        442
Team 2        442
Team 4        441
Team 10       439
Team 9        438
Team 22       438
Team 13       436
Team 18       436
Team 19       436
Team 21       436
Team 7        435
Team 16       435
Team 5        434
Team 24       433
Team 1        433
Team 23       432
Team 12       431
Team 15       431
Team 17       430
Team 11       430
Team 20       430
Team 3        427
Team 8        426
unassigned     83
Name: count, dtype: int64
print("\nNegatif Dönem Record Comptype Dağılımı:")

Negatif Dönem Record Comptype Dağılımı:
print(neg_period['record_comptype'].value_counts())
record_comptype
Presence       4887
Efficacy       4887
Absence         186
Sacrifice       102
Teamwork         96
Feat             87
Idea             87
Slip             33
Lapse            29
Disruption       12
Sabotage         11
Termination       4
Onboarding        4
Name: count, dtype: int64
print("\nPozitif Dönem Vardiya Dağılımı:")

Pozitif Dönem Vardiya Dağılımı:
print(pos_period['sub_shift'].value_counts())
sub_shift
Shift 3       3556
Shift 2       3544
Shift 1       3531
unassigned      20
Name: count, dtype: int64
print("\nPozitif Dönem Takım Dağılımı:")

Pozitif Dönem Takım Dağılımı:
print(pos_period['sub_team'].value_counts())
  
sub_team
Team 17       455
Team 7        448
Team 10       448
Team 16       446
Team 22       444
Team 19       444
Team 14       444
Team 15       443
Team 1        442
Team 23       442
Team 8        440
Team 13       439
Team 24       439
Team 6        439
Team 2        438
Team 20       438
Team 18       438
Team 11       437
Team 3        437
Team 5        435
Team 9        435
Team 21       435
Team 4        432
Team 12       430
unassigned     83
Name: count, dtype: int64
print("\nPozitif Dönem Record Comptype Dağılımı:")
  

Pozitif Dönem Record Comptype Dağılımı:
print(pos_period['record_comptype'].value_counts())
  
record_comptype
Presence       4913
Efficacy       4913
Absence         158
Teamwork        145
Sacrifice       138
Idea             93
Feat             77
Lapse            35
Disruption       35
Slip             26
Sabotage          5
Termination       3
Onboarding        3
Name: count, dtype: int64
>>> import matplotlib.pyplot as plt
>>> daily_pred = df.groupby('event_date')['recorded_efficacy'].mean()
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> daily_pred.plot(marker='o', color='purple')
<Axes: xlabel='event_date'>
>>> plt.axhline(0, color='gray', linestyle='--')
<matplotlib.lines.Line2D object at 0x0000018B4F641810>
>>> plt.title("Modelin Günlük Ortalama Tahmin Değerleri (Recorded Efficacy)")
Text(0.5, 1.0, 'Modelin Günlük Ortalama Tahmin Değerleri (Recorded Efficacy)')
>>> plt.xlabel("Tarih")
Text(0.5, 0, 'Tarih')
>>> plt.ylabel("Tahmin Değeri")
Text(0, 0.5, 'Tahmin Değeri')
>>> plt.grid(True)
>>> plt.tight_layout()
>>> plt.show()
>>> daily_df = df.groupby('event_date')[['recorded_efficacy', 'actual_efficacy_h']].mean()
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> plt.plot(daily_df.index, daily_df['recorded_efficacy'], label='Tahmin (recorded_efficacy)', marker='o')
[<matplotlib.lines.Line2D object at 0x0000018B698ED1D0>]
>>> plt.plot(daily_df.index, daily_df['actual_efficacy_h'], label='Gerçek (actual_efficacy_h)', marker='x')
[<matplotlib.lines.Line2D object at 0x0000018B698ED310>]
>>> plt.axhline(0, color='gray', linestyle='--')
<matplotlib.lines.Line2D object at 0x0000018B698ED450>
>>> plt.title("Model Tahmini vs Gerçek Değerler Zaman Serisi")
Text(0.5, 1.0, 'Model Tahmini vs Gerçek Değerler Zaman Serisi')
>>> plt.xlabel("Tarih")
Text(0.5, 0, 'Tarih')
>>> plt.ylabel("Efficacy Değeri")
Text(0, 0.5, 'Efficacy Değeri')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x0000018B4F5CB0E0>
>>> plt.grid(True)
>>> plt.tight_layout()
>>> plt.show()
