from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
from adam import AdamOptimizer
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/adaptive-moment-estimation")
def calculo(samples: int, n_features: int, lr: float, beta1: float, beta2: float, epsilon: float):
    output_file = 'adam.png'
    
    # Datos de ejemplo
    X = np.linspace(-5, 5, samples)
    y = 2 * X ** 2 + 3 * X + 5 + np.random.randn(*X.shape) * 2

    def loss_function(w, X, y):
        y_pred = w[0] * X**2 + w[1] * X + w[2]
        return np.mean((y - y_pred)**2)

    # Inicialización del optimizador
    num_features=n_features
    #lr=0.1
    #beta1=0.9
    #beta2=0.999
    #epsilon=1e-8

    adam = AdamOptimizer(num_features, lr, beta1, beta2, epsilon)

    # Entrenamiento
    losses = []
    for _ in range(1000):
        weights = adam.get_weights()
        gradients = [
            -2 * np.mean((y - (weights[0] * X**2 + weights[1] * X + weights[2])) * X**2),
            -2 * np.mean((y - (weights[0] * X**2 + weights[1] * X + weights[2])) * X),
            -2 * np.mean(y - (weights[0] * X**2 + weights[1] * X + weights[2]))
        ]
        adam.update(gradients)
        losses.append(loss_function(weights, X, y))

    # Gráfica de dispersión de datos reales y predicción
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, label="Datos reales")
    plt.plot(X, weights[0] * X**2 + weights[1] * X + weights[2], color='red', label="Predicción")
    plt.title("Dispersión de datos")
    plt.legend()

    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(losses, label="Pérdida")
    plt.title("Curva de pérdida")
    plt.xlabel("Iteraciones")
    plt.ylabel("Pérdida")
    plt.legend()

    plt.tight_layout()
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/adaptive-moment-estimation-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
