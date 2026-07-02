#Codigo desenvolvido p/ treinamento de dados meteorologicos p/ auxilo de predicao
#de ocorrencia de chuva

import numpy as num
import pandas as pd
import matplotlib as mat
import sklearn as skl
import seaborn as smn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformerr, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#CARREGAMENTO de dados
url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/_0eYOqji3unP1tDNKWZMjg/weatherAUS-2.csv"
df = pd.read_csv(url)
df.head()
df.count()

#REMOCAO de linhas com valores ausentes
df = df.dropna()
df.info()

#VISUALIZACAO colunas
df.columns()

#RENOMEAR colunas
df = df.rename(columns={'RainToday': 'RainYesterday',
                        'RainTomorrow': 'RainToday'
                        })

#SELACAO DE LOCAIS ()dados
df = df[df.Location.isin(['Melbourne','MelbourneAirport','Watsonia',])]
df. info()

#CRIAR uma funcao de estacoes para dados SAZONAIS
def date_to_season(date):
    month = date.month
    if (month == 12) or (month == 1) or (month == 2):
        return 'Summer'
    elif (month == 3) or (month == 4) or (month == 5):
        return 'Autumn'
    elif (month == 6) or (month == 7) or (month == 8):
        return 'Winter'
    elif (month == 9) or (month == 10) or (month == 11):
        return 'Spring'

# Converter a coluna 'Data' para o formato de data e hora.
df['Date'] = pd.to_datetime(df['Date'])

# Aplique a função à coluna 'Data'.
df['Season'] = df['Date'].apply(date_to_season)

df=df.drop(columns=['Date'])
df.show()

# Usamos .drop para remover a coluna de destino do conjunto de entrada
#X = df.drop(columns='RainToday', axis=1) ERROR(PYTHON VERSION)
X = df.drop(columns='RainToday')
y = df['RainToday']

#EQUILIBRAR classes
y.value_counts()

#Divisao de dados - ESTRATIFICACAO
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.2, stratify= y, random_state=42)

#transformadores de PRE-PROCESSAMENTO para recursos numéricos e categóricos
numeric_features = X_train.select_dtypes(include=['float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

# Dimensionar as características numéricas
numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])

# Codifique as variáveis categóricas usando one-hot encoding.
categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

#[16]. Combinar transformadores em um unico transformador de coluna
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

