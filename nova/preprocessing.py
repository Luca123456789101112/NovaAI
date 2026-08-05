import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, clone


def make_num_pipeline():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ])


class CustomOneHotEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, handle_unknown="ignore", dtype_include=("object", "category")):
        self.handle_unknown = handle_unknown
        self.dtype_include = dtype_include

    def fit(self, X, y=None):
        self._cat_columns = list(X.select_dtypes(include=list(self.dtype_include)).columns)
        self._oh = OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
        self._oh.fit(X[self._cat_columns])
        self._encoded_columns = self._oh.get_feature_names_out(self._cat_columns)
        return self

    def transform(self, X, y=None):
        missing = set(self._cat_columns) - set(X.columns)
        if missing:
            raise ValueError(f"Faltan columnas categóricas esperadas: {missing}")

        X_cat = X[self._cat_columns]
        X_num = X.drop(columns=self._cat_columns)

        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh = pd.DataFrame(
            X_cat_oh,
            columns=self._encoded_columns,
            index=X.index,
        )
        return pd.concat([X_num, X_cat_oh], axis=1)

    def get_feature_names_out(self, input_features=None):
        input_features = [] if input_features is None else list(input_features)
        num_columns = [c for c in input_features if c not in self._cat_columns]
        return num_columns + list(self._encoded_columns)


class DataFramePreparer(BaseEstimator, TransformerMixin):

    def __init__(self, num_pipeline=None, cat_encoder=None):
        self.num_pipeline = num_pipeline
        self.cat_encoder = cat_encoder

    def fit(self, X, y=None):
        self._num_attribs = list(X.select_dtypes(exclude=['object', 'category']).columns)
        self._cat_attribs = list(X.select_dtypes(include=['object', 'category']).columns)

        num_pipe = clone(self.num_pipeline) if self.num_pipeline is not None else make_num_pipeline()
        cat_enc = clone(self.cat_encoder) if self.cat_encoder is not None else CustomOneHotEncoder()

        transformers = []
        if self._num_attribs:
            transformers.append(("num", num_pipe, self._num_attribs))
        if self._cat_attribs:
            transformers.append(("cat", cat_enc, self._cat_attribs))

        if not transformers:
            raise ValueError("El DataFrame no tiene columnas numéricas ni categóricas.")

        self._full_pipeline = ColumnTransformer(transformers, verbose_feature_names_out=False)
        self._full_pipeline.set_output(transform="default")
        self._full_pipeline.fit(X)
        return self

    def transform(self, X, y=None):
        X_prep = self._full_pipeline.transform(X)
        columns = self._full_pipeline.get_feature_names_out()
        return pd.DataFrame(X_prep, columns=columns, index=X.index)

    def get_feature_names_out(self, input_features=None):
        return self._full_pipeline.get_feature_names_out(input_features)


def drop_rows_with_missing_target(X, y):
    """Elimina filas donde el target es NaN. NO tocar las features aquí:
    para eso ya está el SimpleImputer dentro del pipeline numérico."""
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.1, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

#Ejemplo de uso:

#Dividir el DataFrame en conjuntos de entrenamiento, validación y prueba
#train_df, val_df, test_df = train_val_test_split(df)

# Separar variables predictoras y variable objetivo
#X_train = train_df.drop(columns=TARGET)
#y_train = train_df[TARGET]

#X_val = val_df.drop(columns=TARGET)
#y_val = val_df[TARGET]

#X_test = test_df.drop(columns=TARGET)
#y_test = test_df[TARGET]

# Eliminar filas cuyo target sea NaN
#X_train, y_train = drop_rows_with_missing_target(X_train, y_train)
#X_val, y_val = drop_rows_with_missing_target(X_val, y_val)
#X_test, y_test = drop_rows_with_missing_target(X_test, y_test)

# Crear el preprocesador
#preprocessor = DataFramePreparer()

# Ajustar el preprocesador con el conjunto de entrenamiento
#preprocessor.fit(X_train)

# Transformar el conjunto de entrenamiento
#X_train_prep = preprocessor.transform(X_train)

# Transformar el conjunto de validación
#X_val_prep = preprocessor.transform(X_val)

# Transformar el conjunto de prueba
#X_test_prep = preprocessor.transform(X_test)

# Mostrar los nombres de las variables resultantes
#print(preprocessor.get_feature_names_out())