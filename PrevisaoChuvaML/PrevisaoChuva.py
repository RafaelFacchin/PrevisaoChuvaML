import numpy as num
import pandas as pd
import matplotlib as mat
import sklearn as skl
import seaborn as smn

#from sklearn.compose import ColumnTransformerr
#from sklearn.pipeline import Pipeline
#from sklearn.preprocessing import StandardScaler, OneHotEncoder
#from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

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
