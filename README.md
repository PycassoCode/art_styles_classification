# Clasificación de estilos artísticos con redes convolucionales

**Aprendizaje Automático II** (curso 2026-27). Clasificador de imágenes de pinturas en 7 estilos artísticos, implementado en PyTorch.

## Objetivo

Entrenar y comparar redes convolucionales que reconozcan el estilo de un cuadro. Se parte de una CNN sencilla como la vista en clase y se ajustan manualmente sus hiperparámetros, documentando el efecto de cada cambio. Como extra se prueba *transfer learning* con una ResNet-50 preentrenada.

## Datos

* **7 clases**: Baroque, Cubism, Impressionism, Medieval, Minimalism, Pointillism y Ukiyo-e.
* **300 imágenes por clase** (2100 en total), organizadas en una carpeta por clase (formato `ImageFolder`).
* Partición estratificada **70 / 15 / 15**: 1470 imágenes de entrenamiento, 315 de validación y 315 de test, con semilla fija (42).
* El aumento de datos se aplica solo al conjunto de entrenamiento.

Las imágenes no se incluyen en el repositorio. El cuaderno espera encontrarlas en Google Drive en `PROYECTO AA2/dataset/<clase>/`.

## Contenido

| Archivo | Descripción |
|---|---|
| `trabajo_aa2.ipynb` | Cuaderno completo: carga de datos, modelos, configuraciones, comparativa y evaluación en test. |
| `resultados_configuraciones.csv` | Tabla con todas las configuraciones y sus métricas (se genera al ejecutar el cuaderno). |
| `README.md` | Este archivo. |

## Cómo ejecutarlo

1. Subir la carpeta `dataset` a Google Drive dentro de `PROYECTO AA2`.
2. Abrir `trabajo1_aa2.ipynb` en Google Colab y activar la GPU (Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU).
3. Ejecutar todas las celdas (Entorno de ejecución → Ejecutar todo). El cuaderno monta Drive, copia las imágenes redimensionadas al disco local de Colab (la primera vez tarda unos minutos) y entrena todas las configuraciones.

Los pesos del mejor modelo se guardan como `mejor_modelo.pt` y la tabla de resultados como CSV en la carpeta del proyecto de Drive.

## Metodología

* **Parada anticipada** sobre la pérdida de validación, restaurando los pesos de la mejor época.
* **Selección de modelos solo con validación**. El conjunto de test se usa una única vez al final.
* **Búsqueda manual (C1 a C10)**: cada configuración parte de una anterior y cambia uno o un grupo coherente de hiperparámetros.

| Config. | Cambio principal |
|---|---|
| C1 | Base: 3 capas convolucionales (16-32-64, kernel 5), 1 capa densa de 128, ReLU, dropout 0.25, Adam, lr 1e-3, lote 32 |
| C2 | Tasa de aprendizaje 1e-4 |
| C3 | Dropout 0.5 |
| C4 | Lote de 64 |
| C5 | Más filtros (32-64-128) |
| C6 | 4 capas convolucionales (32-64-128-256), kernel 3, BatchNorm, 256 neuronas, lr 1e-3, lote 32 |
| C7 | Dos capas densas (512-128) |
| C8 | LeakyReLU |
| C9 | *Global average pooling* |
| C10 | Aumento de datos, reducción del lr en meseta, 60 épocas y paciencia 10 |
| ResNet-50 | *Transfer learning* en dos fases: cabeza congelada y ajuste fino |

## Autoras

Karla Mederos y Sara Carrizosa, ©PycassoCode
