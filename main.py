import numpy as np
from sklearn import datasets

# Generar los datos de entrenamiento
X, Y = datasets.make_regression(n_samples=50, n_features=1, noise=20)

# Hiperparámetros del algoritmo
tasa_aprendizaje = 0.01
num_iteraciones = 100

# Inicializar los parámetros de la regresión lineal
intercepto_actual = 0
pendiente_actual = 0

# Crear el archivo de texto para guardar los resultados
archivo_resultados = open("regresion_resultados.txt", "w")

# Implementar el algoritmo de regresión lineal por gradiente descendente
for iteracion in range(num_iteraciones):
    # Calcular la derivada de la función de costo con respecto a los parámetros
    derivada_intercepto = -(2/len(X)) * np.sum(Y - (pendiente_actual*X + intercepto_actual))
    derivada_pendiente = -(2/len(X)) * np.sum(X * (Y - (pendiente_actual*X + intercepto_actual)))

    # Actualizar los parámetros usando la tasa de aprendizaje y las derivadas
    nuevo_intercepto = intercepto_actual - tasa_aprendizaje * derivada_intercepto
    nueva_pendiente = pendiente_actual - tasa_aprendizaje * derivada_pendiente

    # Guardar los valores de la iteración en el archivo de resultados
    archivo_resultados.write("Iteración: {}\n".format(iteracion))
    archivo_resultados.write("Puntos:\n")
    for i in range(len(X)):
        archivo_resultados.write("({}, {})\n".format(X[i][0], Y[i]))
    archivo_resultados.write("Tasa de aprendizaje: {}\n".format(tasa_aprendizaje))
    archivo_resultados.write("Anterior intercepto: {}\n".format(intercepto_actual))
    archivo_resultados.write("Anterior pendiente: {}\n".format(pendiente_actual))
    archivo_resultados.write("Derivada (intercepto): {}\n".format(derivada_intercepto))
    archivo_resultados.write("Derivada (pendiente): {}\n".format(derivada_pendiente))
    archivo_resultados.write("Nuevo intercepto: {}\n".format(nuevo_intercepto))
    archivo_resultados.write("Nueva pendiente: {}\n".format(nueva_pendiente))
    archivo_resultados.write("\n")

    # Actualizar los parámetros para la siguiente iteración
    intercepto_actual = nuevo_intercepto
    pendiente_actual = nueva_pendiente

# Cerrar el archivo de resultados
archivo_resultados.close()