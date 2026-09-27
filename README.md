# NovaAI

Librería de redes neuronales implementada desde cero en Python con NumPy, con fines educativos. Incluye los componentes básicos de un framework de deep learning (capas, funciones de activación, funciones de pérdida y un modelo secuencial), además de utilidades de preprocesamiento y notebooks de ejemplo aplicados a un caso real.

## Estructura del proyecto

```
NovaAI/
├── nova/                  # Librería principal
│   ├── __init__.py
│   ├── layers.py          # Dense, Dropout
│   ├── activations.py     # ReLu, Softmax, Tanh, Sigmoid
│   ├── losses.py          # BinaryCrossEntropy, CategoricalCrossEntropy
│   ├── models.py          # Modelo Sequential (forward / backward / fit / predict)
│   └── preprocessing.py   # Utilidades de preprocesamiento (pandas / scikit-learn)
├── Examples/               # Notebooks de ejemplo
│   ├── v1/
│   └── v2/
├── requirements.txt
└── README.md
```

## Instalación

```bash
pip install -r requirements.txt
```

## Uso básico

```python
from nova import Sequential, Dense, ReLu, Sigmoid, BinaryCrossEntropy

model = Sequential()
model.add(Dense(input_dim=10, output_dim=16, initializer="he"))
model.add(ReLu())
model.add(Dense(input_dim=16, output_dim=1, initializer="he"))
model.add(Sigmoid())

model.compile(loss=BinaryCrossEntropy())
model.fit(X_train, y_train, iterations_num=5000)

predictions = model.predict(X_test)
```

## Ejemplos

La carpeta `Examples/` contiene notebooks que aplican la librería, y el módulo `preprocessing`, a un dataset real de voluntariado (`dataset_voluntariado.csv`). Actualmente conviven dos versiones (`v1` y `v2`); conviene revisar cuál es la vigente y archivar o eliminar la otra.

## Estado

Proyecto personal en desarrollo, orientado al aprendizaje de los fundamentos del deep learning.
