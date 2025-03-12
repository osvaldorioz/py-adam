
ADAM (Adaptive Moment Estimation) es un optimizador ampliamente utilizado en aprendizaje automático debido a su estabilidad y eficiencia. Combina las ventajas de dos algoritmos populares:  
- **Momentum:** Que acelera la convergencia en gradientes suaves.  
- **RMSProp:** Que adapta la tasa de aprendizaje a cada parámetro individualmente.  

El algoritmo ADAM realiza estos pasos clave:

![imagen](https://github.com/user-attachments/assets/9978f94f-be98-4aac-b01e-44b43ca084e6)


---

#### **Implementación en C++ con Eigen**
El código sigue fielmente la lógica del algoritmo ADAM utilizando la biblioteca Eigen para las operaciones vectoriales.

- **Inicialización:**  
  ```cpp
  weights = VectorXd::Zero(num_features);
  m = VectorXd::Zero(num_features);
  v = VectorXd::Zero(num_features);
  ```
  Aquí se crean los vectores `weights`, `m` y `v` usando Eigen.

- **Actualización de momentos:**  
  ```cpp
  m = beta1 * m + (1 - beta1) * g;
  v = beta2 * v + (1 - beta2) * g.array().square().matrix();
  ```
  Se actualizan los momentos según el algoritmo ADAM.

- **Corrección de sesgo:**  
  ```cpp
  VectorXd m_hat = m / (1 - std::pow(beta1, t));
  VectorXd v_hat = v / (1 - std::pow(beta2, t));
  ```
  Se corrige el sesgo introducido por la inicialización en cero.

- **Actualización de pesos:**  
  ```cpp
  weights.array() -= alpha * m_hat.array() / (v_hat.array().sqrt() + epsilon).array();
  ```
  Se utiliza `.array()` para realizar operaciones elemento a elemento, ya que Eigen requiere este modo para cálculos de matrices con escalares.

---

#### **Ventajas del uso de Eigen**
- Operaciones optimizadas para cálculos vectorizados.
- Manejo eficiente de operaciones matriciales y algebraicas.
- Código limpio y fácil de leer, ya que Eigen permite manipular vectores como si fueran arreglos.

---
