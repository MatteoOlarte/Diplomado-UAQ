import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Leer el dataset
data = pd.read_csv('semana 1/data/Titanic - train.csv')

# El atributo 'Age', Cabin y Embarked tienen valores faltantes que debemos procesar
f, ax = plt.subplots(1, 2, figsize=(20, 10))
data['Survived'].value_counts().plot.pie(autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Survived')
sns.countplot(data=data, x='Survived', ax=ax[1])
ax[1].set_title('Survived')
plt.show()

# Hay tres tipos de datos: categóricos, ordinales y continuos (numéricos)

# --------------------------------------------------------------------------------------------------
# Datos Categóricos
# --------------------------------------------------------------------------------------------------

# Dato: "SEX", tiene influencia en la supervivencia
data.groupby(['Sex', 'Survived'])['Survived'].count()
f, ax = plt.subplots(1, 2, figsize=(20, 10))
data[['Sex', 'Survived']].groupby(['Sex']).mean().plot.bar(ax=ax[0])
sns.countplot(data=data, x='Sex', hue='Survived', ax=ax[1])
plt.show()

# Dato: "Pclass", tine influencia en la supervivencia
data.groupby(['Pclass', 'Survived'])['Survived'].count()
f, ax = plt.subplots(1, 3, figsize=(20, 10))
data['Pclass'].value_counts().plot.bar(ax=ax[0])
data[['Pclass', 'Survived']].groupby(['Pclass']).mean().plot.bar(ax=ax[1])
sns.countplot(data=data, x='Pclass', hue='Survived', ax=ax[2])
plt.show()

# --------------------------------------------------------------------------------------------------
# Datos Ordinales
# --------------------------------------------------------------------------------------------------

sns.catplot(data=data, x='Pclass', y='Survived', hue='Sex', kind='point')

# --------------------------------------------------------------------------------------------------
# Datos Continuos
# --------------------------------------------------------------------------------------------------

# Dato: "Age"
print(f'{data['Age'].min()} años')
print(f'{data['Age'].mean()} años')
print(f'{data['Age'].max()} años')

f, ax = plt.subplots(1, 2, figsize=(20, 10))
sns.violinplot(data=data, x='Pclass', y='Age', hue='Survived', split=True, ax=ax[0])
sns.violinplot(data=data, x='Sex', y='Age', hue='Survived', split=True, ax=ax[1])

# --------------------------------------------------------------------------------------------------
# Manejar los datos faltantes para 'Age'
# --------------------------------------------------------------------------------------------------

# Se va a extraer los prefijos de cada uno de los nombres y en base a estos se va a imputar un estimado de la
# edad faltante
prefixes = data['Name'].str.extract(r'([A-Za-z]+)\.')[0]
data['Prefix'] = prefixes

pd.crosstab(data['Prefix'], data['Sex'])

# Se va a simplificar el número de prefijos, renombrando aquellos que no son muy comunes o tienen una frecuencia baja
# en base al contexto de su prefijo se le asignará uno nuevo dentro de las categorías 'Mr', 'Miss', 'Mrs' u 'Other'
data['Prefix'] = data['Prefix'].replace(
    ['Capt', 'Col', 'Countess', 'Don', 'Jonkheer', 'Lady', 'Major', 'Mlle', 'Mme', 'Ms', 'Rev', 'Sir', 'Dr'],
    ['Mr', 'Mr', 'Other', 'Mr', 'Other', 'Other', 'Mr', 'Miss', 'Mrs', 'Mrs', 'Mr', 'Mr', 'Mr'],
)


data.groupby(['Prefix'])['Age'].mean()

data.loc[(data['Age'].isnull()) & (data['Prefix'] == 'Master'), 'Age'] = 4.57
data.loc[(data['Age'].isnull()) & (data['Prefix'] == 'Miss'), 'Age'] = 21.80
data.loc[(data['Age'].isnull()) & (data['Prefix'] == 'Mr'), 'Age'] = 33.02
data.loc[(data['Age'].isnull()) & (data['Prefix'] == 'Mrs'), 'Age'] = 35.80

data.isnull().any()

f, ax = plt.subplots(1, 2, figsize=(20, 10))
data[data['Survived'] == 0].Age.plot.hist(ax=ax[0], bins=100)
ax[0].set_xticks(range(0, 80, 5))
data[data['Survived'] == 1].Age.plot.hist(ax=ax[1], bins=100)
ax[1].set_xticks(range(0, 80, 5))
