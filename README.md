# Clasificación de estilos artísticos con redes convolucionales 

Práctica 1 de **Aprendizaje Automático II** (2026-27), Universidad de Las Palmas de Gran Canaria.
Autoras: Sara Carrizosa y Karla Mederos.

Clasificador de pinturas en 7 estilos (Baroque, Cubism, Impressionism, Medieval, Minimalism, Pointillism y Ukiyo) con PyTorch. Se comparan 10 configuraciones de una CNN propia y una ResNet-50 preentrenada (*transfer learning*).

## Resultados

| Modelo | Acc. validación | Acc. test |
|---|---|---|
| Mejor CNN propia (C10) | 74,29 % | 74,60 % |
| ResNet-50 (*transfer learning*) | 92,70 % | 87,62 % |

## Contenido del repositorio

| Ruta | Descripción |
|---|---|
| `trabajo_aa2.ipynb` | Cuaderno completo: datos, modelos, las 10 configuraciones, ResNet-50, comparativa y evaluación en test. |
| `memoria/` | Memoria en LaTeX (`memoria.tex`) y sus figuras. |
| `resultados_configuraciones.csv` | Tabla de todas las configuraciones (se genera al ejecutar el cuaderno). |

## Cómo reproducirlo

El cuaderno está pensado para **Google Colab con GPU** (Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU). Todas las pruebas usan la semilla 42 y la misma partición estratificada 70/15/15, de modo que los resultados deberían ser muy parecidos (la GPU puede introducir pequeñas diferencias).

### Pasos

Las imágenes no están en el repositorio (pesan demasiado). Están en Google Drive: **[ENLACE_AQUÍ](https://drive.google.com/drive/folders/1-5wu8q_eYcUN349utM9XhduKHMM2jdvo?usp=sharing)**. Es una carpeta `dataset` con las 2100 imágenes (300 por clase, una subcarpeta por estilo, formato `ImageFolder`).

1. Descargad la carpeta del enlace (Drive la descarga como un zip) y descomprimidla.
2. En vuestro Google Drive, cread una carpeta `PROYECTO AA2` y subid dentro la carpeta `dataset`, de modo que quede `Mi unidad/PROYECTO AA2/dataset/<estilo>/imagen.jpg` con las 7 subcarpetas: `Baroque`, `Cubism`, `Impressionism`, `Medieval`, `Minimalism`, `Pointillism` y `Ukiyo`.
3. Abrid `trabajo1_aa2.ipynb` en Google Colab, activad la GPU y ejecutad todas las celdas (Entorno de ejecución → Ejecutar todo). El cuaderno monta Drive, lee las imágenes de `PROYECTO AA2/dataset`, las copia reducidas al disco local de Colab, entrena todas las configuraciones y evalúa en test.

El cuaderno guarda en `PROYECTO AA2` de Drive la tabla `resultados_configuraciones.csv` y los pesos del mejor modelo (`mejor_modelo.pt`).

## Metodología

* Partición estratificada: 1470 imágenes de entrenamiento, 315 de validación y 315 de test.
* Parada anticipada sobre la pérdida de validación, restaurando los pesos de la mejor época.
* Los modelos se eligen solo con validación; el test se usa una única vez al final.
* Búsqueda manual de hiperparámetros: cada configuración cambia uno (o un grupo coherente) respecto a una anterior.

| Config. | Cambio principal |
|---|---|
| C1 | Base: 3 conv (16-32-64, kernel 5), 1 capa densa de 128, ReLU, dropout 0,25, Adam, lr 1e-3, lote 32 |
| C2 | lr 1e-4 |
| C3 | Dropout 0,5 |
| C4 | Lote de 64 |
| C5 | Más filtros (32-64-128) |
| C6 | 4 conv (32-64-128-256), kernel 3, BatchNorm, 256 neuronas, lr 1e-3, lote 32 |
| C7 | Dos capas densas (512-128) |
| C8 | LeakyReLU |
| C9 | *Global average pooling* |
| C10 | Aumento de datos, reducción del lr en meseta, 60 épocas y paciencia 10 |
| ResNet-50 | *Transfer learning* en dos fases: capa final y ajuste fino |

## Datos

300 imágenes por clase, descargables desde el enlace de arriba. Origen de las imágenes: **completar** (por ejemplo, WikiArt/Kaggle y las fuentes del estilo Medieval). Comprobad la licencia de las imágenes antes de hacer público el repositorio.
