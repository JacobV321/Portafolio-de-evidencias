#Jacob Valdenegro Monzón A01640992

import numpy as np
import statsmodels.api as sm

# Datos proporcionados
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
Y = np.array([10.06, 6.6, 10.91, 17.96, 18.47, 9.09, 18.8, 16.39, 18.59, 22.64, 
              23.58, 30.82, 30.04, 29.49, 32.78, 34.33, 40.98, 36.18, 40.25, 37.58])

# Paso 1: Calculo de los coeficientes
n = len(X)
X_mean = np.mean(X)
Y_mean = np.mean(Y)

# Formulas de minimos cuadrados ordinarios
beta_1 = np.sum((X - X_mean) * (Y - Y_mean)) / np.sum((X - X_mean)**2)
beta_0 = Y_mean - beta_1 * X_mean

# Resultados del calculo
print("Coeficientes calculados:")
print(f"Intercepto (beta_0): {beta_0:.4f}")
print(f"Pendiente (beta_1): {beta_1:.4f}")

# Construccion de la ecuacion de regresion
print("\nEcuacion de la regresion lineal:")
print(f"Y = {beta_0:.4f} + {beta_1:.4f} * X")

# Paso 2: Verificacion utilizando statsmodels
X_with_const = sm.add_constant(X)
model = sm.OLS(Y, X_with_const).fit()

print("\nResumen del modelo (statsmodels):")
print(model.summary())
