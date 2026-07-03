#Codigo desenvolvido p/ treinamento de dados meteorologicos p/ auxilo de predicao
#de ocorrencia de chuva


import numpy as num
import pandas as pd
import matplotlib as mat
import sklearn as skl
import seaborn as smn
import plt
import sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformerr, ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot as plt

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

# -->> Combinar transformadores em um unico transformador de coluna
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# -->> pipeline combinando o pré-processamento com um classificador RANDOM FOREST.
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# -->> Grade de parâmetros com validação cruzada:
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# -->> Validação cruzada por busca em grade
cv = StratifiedKFold(n_splits=5, shuffle=True)

# -->> Ajuste do GridSearchCV ao pipeline
grid_search = GridSearchCV(pipeline, param_grid, cv = cv, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# -->> Melhores parâmetros e a melhor pontuação de validação cruzada
print("\nBest parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# -->> Pontuação estimada do modelo
test_score = grid_search.score(X_test, y_test)
print("Test set score: {:.2f}".format(test_score))

# -->> Previsões do modelo a partir do estimador de busca em grade nos dados não vistos
y_pred = grid_search.predict(X_test)

# -->> Relatório de classificação
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -->> Matriz de confusão
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

# -->> Extracao das características
feature_importances = grid_search.best_estimator_['classifier'].feature_importances

# -->> extracao das importância/ características; e Representacao em um gráfico de barras.
# Combinar nomes de recursos numéricos e categóricos
feature_names = numeric_features + list(grid_search.best_estimator_['preprocessor']
                                        .named_transformers_['cat']
                                        .named_steps['onehot']
                                        .get_feature_names_out(categorical_features))

feature_importances = grid_search.best_estimator_['classifier'].feature_importances_

importance_df = pd.DataFrame({'Feature': feature_names,
                              'Importance': feature_importances
                             }).sort_values(by='Importance', ascending=False)

N = 20  # Altere este número para exibir mais ou menos recursos.
top_features = importance_df.head(N)

# Plotando
plt.figure(figsize=(10, 6))
plt.barh(top_features['Feature'], top_features['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # Inverta o eixo y para mostrar a característica mais importante no topo.
plt.title(f'Top {N} Most Important Features in predicting whether it will rain today')
plt.xlabel('Importance Score')
plt.show()

# -->> Comparar os resultados com o modelo anterior:
print(classification_report(y_test, y_pred))

# Gerar a matriz de confusao
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d')

# Defina o titulo e os rotulos
plt.title('Titanic Classification Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Mostre o plote
plt.tight_layout()
plt.show()

