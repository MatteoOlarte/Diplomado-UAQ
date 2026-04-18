import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Leer el dataset
df = pd.read_csv('data/winemag-data-130k-v2.csv')

# ==================================================================
# Punto 1
# ==================================================================
# Observaciones: Distribución unimodal centrada en 88 puntos, con pocos vinos en 100.
m = df['points'].mode()[0]
print('Moda:', m)
print('Puntaje mínimo:', df['points'].min())
print('Puntaje máximo:', df['points'].max())

df['points'].hist(bins=20, edgecolor='black')
plt.title('Distribución de Calificaciones de Vinos')
plt.xlabel('Puntaje')
plt.ylabel('Frecuencia')
plt.axvline(x=m, color='red', linestyle='--', label=f'Moda = {m}')
plt.legend()
plt.show()

# ==================================================================
# Punto 2
# ==================================================================
# Observaciones: Hay outliers altos. Decisión: eliminar precios > Q3 + 1.5*IQR
prices = df['price'].dropna()

# Boxplot rápido para ver outliers
sns.boxplot(x=prices)
plt.title('Boxplot precios - con outliers')
plt.show()

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
df_clean = df[df['price'] <= Q3 + 1.5 * IQR]

sns.histplot(df_clean['price'], kde=True)
plt.title('Distribución de precios sin outliers')
plt.show()

# ==================================================================
# Punto 3
# ==================================================================
# Observaciones: Top 20 países con mejor puntaje promedio (usando df_clean)
country_stats = df_clean.groupby('country')[['price', 'points']].mean().sort_values(by='points', ascending=False).head(20)
print(country_stats)

sns.barplot(x=country_stats.index, y=country_stats['points'], palette='viridis')
plt.title('Top 20 Países - Puntaje Promedio')
plt.xticks(rotation=45)
plt.show()

# ==================================================================
# Punto 4
# ==================================================================
# Observaciones: Relación entre las variedades más comunes y su calidad
top_varieties = df_clean['variety'].value_counts().nlargest(10).index
df_top = df_clean[df_clean['variety'].isin(top_varieties)]

sns.boxplot(x='variety', y='points', data=df_top, palette='viridis')
plt.title('Boxplot: Variedad vs Points (Top 10)')
plt.xticks(rotation=45)
plt.show()

# ==================================================================
# Punto 5
# ==================================================================
country_avg_points = df_clean.groupby('country')['points'].transform('mean')
df_clean.loc[:, 'score points'] = df_clean['points'] / country_avg_points

# ==================================================================
# Punto 6
# ==================================================================
# Estrategia: Rellenar con promedio de precio por país (mantiene consistencia por país)
country_avg_price = df_clean.groupby('country')['price'].transform('mean')
df_clean['price'] = df_clean['price'].fillna(country_avg_price)
df_clean['price'] = df_clean['price'].fillna(df_clean['price'].mean())
print("Faltantes en price después de imputar:", df_clean['price'].isnull().sum())

# ==================================================================
# Punto 7
# ==================================================================
tropical = df['description'].str.contains('tropical', case=False, na=False).sum()
fruity = df['description'].str.contains('fruity', case=False, na=False).sum()
print(f"Tropical aparece {tropical} veces")
print(f"Fruity aparece {fruity} veces")

# ==================================================================
# Punto 8
# ==================================================================
# Transformar country a numérico (label encoding simple)
df_clean['country_code'] = df_clean['country'].astype('category').cat.codes

# ==================================================================
# Punto 9
# ==================================================================
def get_stars(x):
    if x >= 95:
        return 3
    elif x >= 85:
        return 2
    else:
        return 1

df_clean['stars'] = df_clean['points'].apply(get_stars)
