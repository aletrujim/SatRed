## Uso de Métodos de Aprendizaje Automático y teledetección para clasificación de uso y cobertura del suelo en un valle semiárido de la Patagonia

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Tabla de Contenido</summary>
  <ol>
    <li>
     <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#results">Results</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

![virch](https://github.com/aletrujim/SatRed/blob/main/compare-classifiers/images/fig1.jpg)

Comparamos 7 métodos tradicionales de aprendizaje automático supervisado aplicados a la clasificación del uso y la cobertura del suelo a partir de imágenes satelitales de Sentinel-2 y de datos adquiridos sobre el terreno. El sitio de estudio es el valle agrícola-ganadero en la Cuenca  Inferior del Río Chubut que tiene una extensión de 225 km2 y está situado en la Patagonia semiárida oriental argentina.

<!-- GETTING STARTED -->
## Getting Started

Todos los procedimientos se llevaron a cabo en una máquina virtual (Microsoft Azure 2), con el sistema operativo Windows Server 2016 Datacenter, de tamaño estándar NC6 (6 vCPUs [Intel Xeon CPU E5-2690 v3 2,60 GHz], 56 GB de memoria RAM y un co-procesador GPU NVIDIA Tesla K80). Los algoritmos se implementaron utilizando el lenguaje de programación Python. Se utilizó el paquete Rasterio para acceder y procesar los datos ráster geoespaciales, y Shapely para la manipulación de polígonos. 

Para correr una copia local de este proyecto siga estos sencillos pasos.

### Prerequisites

* GDAL
  ```sh
  pip install GDAL
  ```
* scikit-learn
  ```sh
  pip install scikit-learn
  ```
  
 ### Installation

1. Descargar las imagenes de entrenamiento y validación [Link](https://drive.google.com/drive/folders/1Y68DvzrQ0ahoN0zNH-h17x6bjXxJqXim?usp=sharing) o preparar las suyas. Si usa las imagenes descargadas tenga en cuenta que fueron preparadas y particionadas con [Partition](https://github.com/aletrujim/SatRed/tree/main/partition)
3. Clone the repo
   ```sh
   git clone https://github.com/aletrujim/SatRed.git
   ```
3. Install packages
   ```sh
   pip install -r requirements.txt
   ```
4. Run python script
   ```sh
   python compare-classifiers.py --train=train --test=test --segmented=result
   ```
   
<!-- RESULTS -->
## Results

<!-- CONTACT -->
## Contact

Alexa Trujillo - [@aletrujim](https://twitter.com/aletrujim)

Project Link: [https://github.com/aletrujim/SatRed](https://github.com/aletrujim/SatRed)
 
