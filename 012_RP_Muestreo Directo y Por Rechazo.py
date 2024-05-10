#Salazar Chavez Cristian Uriel
#21310215       

from scipy.stats import norm

# Función de densidad de probabilidad de la distribución objetivo (normal estándar)
def target_distribution(x):
    return norm.pdf(x, mu, sigma)

# Función de densidad de probabilidad de la distribución de propuesta (uniforme en el intervalo [-3, 3])
def proposal_distribution(x):
    return 1 / 6 if -3 <= x <= 3 else 0

# Realizar muestreo por rechazo
samples_rejection = []
for _ in range(1000):
    # Generar una muestra de la distribución de propuesta
    sample = np.random.uniform(-3, 3)
    
    # Calcular la proporción entre la función de densidad de la distribución objetivo y la distribución de propuesta
    ratio = target_distribution(sample) / proposal_distribution(sample)
    
    # Aceptar o rechazar la muestra con probabilidad proporcional a la proporción calculada
    if np.random.uniform(0, 1) < ratio:
        samples_rejection.append(sample)

# Imprimir las primeras 10 muestras
print("\nMuestras generadas por muestreo por rechazo:")
print(samples_rejection[:10])
