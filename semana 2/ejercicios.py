import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Leer el dataset de winemag
df = pd.read_csv('data/winemag-data-130k-v2.csv')

# Configurar el estilo oscuro de las gráficas
plt.style.use('dark_background')
sns.set_theme(style="darkgrid")

# Configuración adicional para un tema más oscuro
plt.rcParams['figure.facecolor'] = '#1a1a1a'
plt.rcParams['axes.facecolor'] = '#2b2b2b'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['text.color'] = '#ffffff'
plt.rcParams['axes.labelcolor'] = '#ffffff'
plt.rcParams['xtick.color'] = '#ffffff'
plt.rcParams['ytick.color'] = '#ffffff'

# ----------------------------------------------------------------------------------------------------------------------
# Punto 1
# ----------------------------------------------------------------------------------------------------------------------
# Observaciones:

# 1. El histograma muestra una distribución unimodal pues solo hay un solo pico en la distribución de los puntos.
# 2. La distribución esta centrada en el valor de 88 puntos, valor que indica la moda de la distribución.
# 3. El puntaje mínimo es de 80 puntos.
# 4. El puntaje máximo es de 100 puntos, con una frecuencia muy baja en ese extremo lo que puede indicar valores
#    atípicos.
# 5. Aunque la distribución sigue una tendencia unimodal (una sola montaña), presenta una anomalía cerca de los 90
#    puntos que rompe la simetría, lo cual es característico de sesgos en la evaluación humana
# ----------------------------------------------------------------------------------------------------------------------

m = df['points'].mode()

print('Moda:', df['points'].mode()[0])
print('Puntaje mínimo:', df['points'].min())
print('Puntaje máximo:', df['points'].max())

plt.figure(figsize=(9, 5))

df['points'].hist(bins=20, edgecolor='black')

plt.title('Distribución de Calificaciones de Vinos')
plt.xlabel('Puntaje (80 - 100 puntos)')
plt.ylabel('Frecuencia de reseñas')
plt.axvline(x=m.iloc[0], color='red', linestyle='--', label=f'moda={m.iloc[0]}')
plt.legend()
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Punto 2
# ----------------------------------------------------------------------------------------------------------------------
# Observaciones:
# Se puede observar en el diagrama que hay valores atípicos, en este caso los precios de los vinos van desde los 4 USD
# hasta los 3300 USD, indicando una gran variabilidad en los precios de los vinos.

# Conclusiones:
# 1. La mayoría de los vinos tienen precios "bajos" o son de un precio promedio.
# 2. Hay unos datos atípicos los cuales son los precios que superan los 200 USD.

# Como los outliers representan menos de 5% del dataset, se puede considerar eliminar estos datos para obtener una
# mejor visualización de la distribución de precios.
# ----------------------------------------------------------------------------------------------------------------------

prices = df['price'].dropna()

f, ax = plt.subplots(1, 2, figsize=(20, 7))

# Gráfica 1: Sin límite (se observan los outliers)
sns.histplot(prices, bins=100, kde=True, ax=ax[0])
ax[0].set_title('Distribución de Precios (Sin filtro)')
ax[0].set_xlabel('Precio (USD)')
ax[0].set_ylabel('Frecuencia')

# Gráfica 2: Con límite de 200 USD para ver la concentración
sns.histplot(prices, bins=100, kde=True, ax=ax[1])
ax[1].set_xlim(0, 200)
ax[1].set_title('Distribución de Precios (Zoom hasta 200 USD)')
ax[1].set_xlabel('Precio (USD)')
ax[1].set_ylabel('Frecuencia')

plt.tight_layout()
plt.show()

# Gráfica 3: Boxplot para visualizar los outliers
plt.figure(figsize=(10, 5))

# Boxplot con todos los datos de precios
sns.boxplot(x=prices)
plt.title('boxplot de los precios - sin eliminar outliers')
plt.xlabel('Precio (USD)')

plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# Calcular el rango intercuartílico (IQR) para detectar y eliminar los outliers

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

limite_superior = Q3 + 1.5 * IQR
outliers = df[df['price'] > limite_superior]

# Contar el número de filas
num_outliers = len(outliers)
porcentaje = (num_outliers / len(df)) * 100

print(f"Número de vinos atípicos: {num_outliers}")
print(f"Representan el {porcentaje:.2f}% del dataset.")

df_clean = df[df['price'] <= Q3 + 1.5 * IQR]

# Mostrar la nueva distribución de precios sin los outliers

sns.histplot(df_clean['price'], kde=True)
plt.show()

sns.boxplot(x=df_clean['price'])
plt.title('boxplot de los precios - sin eliminar outliers')
plt.xlabel('Precio (USD)')

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Punto 3
# ----------------------------------------------------------------------------------------------------------------------
# Observaciones:
# 1. Se compara los dos datasets, el original y el dataset limpio, para observar si hay cambios en los países con
# los vinos más caros y mejor valorados.

# 2. Al comparar ambos datasets, se observa que no hay grandes cambios en los países de este top 20, incluso los
# 4 primeros países con el vino mas caro tienen el mismo orden en ambos datasets.
# ----------------------------------------------------------------------------------------------------------------------

# Sin limpieza de outliers

country_stats = df.groupby('country')[['price', 'points']].mean().sort_values(by='points', ascending=False)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=country_stats.index[:20],
    y=country_stats['points'][:20],
    palette='viridis'
)

plt.title('Top 20 Países con Mayor Puntaje Promedio (Con Outliers)')
plt.xlabel('País')
plt.ylabel('Puntaje Promedio')
plt.ylim(85, 92)  # Ajuste para notar mejor las diferencias
plt.xticks(rotation=45)

plt.show()

# ----------------------------------------------------------------------------------------------------------------------

# Con limpieza de outliers

country_stats_clean = df_clean.groupby('country')[['price', 'points']].mean().sort_values(by='points', ascending=False)

plt.figure(figsize=(12, 6))

sns.barplot(
    x=country_stats_clean.index[:20],
    y=country_stats_clean['points'][:20],
    palette='viridis'
)

plt.title('Top 20 Países con Mayor Puntaje Promedio (Sin Outliers)')
plt.xlabel('País')
plt.ylabel('Puntaje Promedio')
plt.ylim(85, 92)  # Ajuste para notar mejor las diferencias
plt.xticks(rotation=45)

plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Punto 4
# ----------------------------------------------------------------------------------------------------------------------

top_varieties = df_clean['variety'].value_counts().nlargest(20).index
df_top_varieties = df_clean[df_clean['variety'].isin(top_varieties)]

plt.figure(figsize=(14, 8))
sns.boxplot(
    x='variety',
    y='points',
    data=df_top_varieties,
    palette='viridis',
    boxprops={'edgecolor': 'white'},
    whiskerprops={'color': 'white'},
    capprops={'color': 'white'},
    medianprops={'color': 'white'},
    flierprops={'markeredgecolor': 'white'}
)

plt.title('Relación entre Variedad y Calidad (Top 20 variedades)')
plt.xlabel('Variedad')
plt.ylabel('Puntaje')
plt.xticks(rotation=45)
plt.show()

# ----------------------------------------------------------------------------------------------------------------------
# Punto 5
# ----------------------------------------------------------------------------------------------------------------------

country_avg_points = df_clean.groupby('country')['points'].transform('mean')
df_clean.loc[:, 'score points'] = df_clean['points'] / country_avg_points

# ----------------------------------------------------------------------------------------------------------------------
# Punto 6
# ----------------------------------------------------------------------------------------------------------------------
# Observaciones:
# Para rellenar los valores faltantes en la columna 'price', se ha utilizado el precio promedio por país. Pues como 
# cada pais tiene su rango de precios, se evita que se pierda la consistecia en los datos en cuanto al precio, es 
# decir, la grafia de paises con vinos más caros no se vera afectada en su mayoría.
# ----------------------------------------------------------------------------------------------------------------------

# Calcular el precio promedio por país

country_avg_price = df_clean.groupby('country')['price'].transform('mean')
df_clean['price'] = df_clean['price'].fillna(country_avg_price)
df_clean['price'] = df_clean['price'].fillna(df_clean['price'].mean())
print("Valores faltantes en 'price':", df_clean['price'].isnull().sum())

# ----------------------------------------------------------------------------------------------------------------------
# Punto 7
# ----------------------------------------------------------------------------------------------------------------------

# Definir las palabras a buscar

words_to_find = ['tropical', 'fruity']

word_counts = {}

for word in words_to_find:
    count = df['description'].str.contains(word, case=False).sum()
    word_counts[word] = count
    print(f"La palabra '{word}' aparece {count} veces en las descripciones.")

# ----------------------------------------------------------------------------------------------------------------------
# Punto 8
# ----------------------------------------------------------------------------------------------------------------------

df_clean['country_code'] = df_clean['country'].astype('category').cat.codes

# ----------------------------------------------------------------------------------------------------------------------
# Punto 9
# ----------------------------------------------------------------------------------------------------------------------
# Observaciones:
# SE aplico la normalización min-max a la columna 'points' para escalar los puntajes a un rango de 1 a 5. 
# ----------------------------------------------------------------------------------------------------------------------

min_points = df_clean['points'].min()
max_points = df_clean['points'].max()

df_clean['score_5'] = ((df_clean['points'] - min_points) / (max_points - min_points)) * 4.0 + 1.0

df_clean[['points', 'score_5']].drop_duplicates().sort_values('points').head(40)

