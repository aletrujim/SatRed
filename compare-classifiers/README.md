# compare-classifiers

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

Comparamos 7 métodos tradicionales de aprendizaje automático supervisado aplicados a la clasificación del uso y la cobertura del suelo a partir de imágenes satelitales de Sentinel-2 y de datos adquiridos sobre el terreno. El sitio de estudio es el valle agrícola-ganadero en la Cuenca  Inferior del Río Chubut que tiene una extensión de 225 km2 y está situado en la Patagonia semiárida oriental argentina. Con estos métodos obtuvimos predicciones que superan entre el 70 y 80 % de precisión en la clasificación de cultivos frutales, horticultura, terrenos construidos, arbustales, pasturas y agua. Se proponen mejoras que incluyen incorporar modelos de Aprendizaje Profundo, información de textura e imágenes multitemporales para pequeñas áreas de estudio.

<!-- GETTING STARTED -->
## Getting Started

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

1. Descargar las imagenes de entrenamiento y validación [Link](https://drive.google.com/drive/folders/1xSGD-oiCKO55FVOTxMvQR-ePt-nucC38?usp=sharing) o preparar las suyas. Si usa las imagenes descargadas tenga en cuenta que fueron preparadas y particionadas con [Partition](https://github.com/aletrujim/SatRed/tree/main/partition)
3. Clone the repo
   ```sh
   git clone https://github.com/aletrujim/SatRed.git
   ```
3. Install packages
   ```sh
   npm install XXX
   ```
4. Run python script
   ```sh
   python compare-classifiers.py --train=train --test=test --segmented=result
   ```
   
<!-- RESULTS -->
## Results

Nearest Neighbors mostró las mejores puntuaciones globales en comparación con los otros métodos y los peores rendimientos los obtuvieron AdaBoost y QDA. Al examinar las puntuaciones de Precisión, Cohen's Kappa y Hamming Loss, Random Forest también tuvo un buen rendimiento con valores de precisión superiores al 80%. Sin embargo, cuando se analizó por clases, Random Forest tuvo problemas para reconocer las parcelas de Horticultura y de Cultivos Frutales (es decir, bajo valor de Recall).

![table](https://github.com/aletrujim/SatRed/blob/main/compare-classifiers/images/fig2.png)

![fscore](https://github.com/aletrujim/SatRed/blob/main/compare-classifiers/images/fig3.png)



<!-- CONTACT -->
## Contact

Alexa Trujillo - [@your_twitter](https://twitter.com/aletrujim)

Project Link: [https://github.com/aletrujim/SatRed](https://github.com/aletrujim/SatRed)
 
