# Clasificador de imagenes de enfermedades Early Blight y Late Blight en las Papas

Este proyecto emplea técnicas de aprendizaje profundo para clasificar imágenes de patatas. Utiliza augmentacion de datos para balancear el conjunto de datos y utiliza Transfer Learning con la red neuronal convolucional InceptionV3 para clasificar las imágenes.

- dataAugmentation.py: Este script se utiliza para aumentar el conjunto de datos mediante la generación de imágenes transformadas.
- TPIAInception.py: Este script contiene el modelo de red neuronal convolucional InceptionV3 para clasificar las imágenes.
- TPIAReducido.py: Este script contiene un modelo convolucional reducido para clasificar las imágenes.
- best_model.pth: Este archivo contiene los pesos del modelo que ha obtenido los mejores resultados hasta ahora.