## SatRed v.2: Modelo de clasificación de uso/cobertura del suelo basado en imágenes satelitales multiespectrales y redes neuronales aplicado a un valle semiárido de la Patagonia

![satred](https://github.com/aletrujim/SatRed/blob/main/images/satred_arquitectura.png)

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Tabla de Contenido</summary>
  <ol>
    <li>
     <a href="#about-the-project">Proyecto</a>
    </li>
    <li>
      <a href="#getting-started">Cómo empezar</a>
      <ul>
        <li><a href="#prerequisites">Requisitos previos</a></li>
        <li><a href="#installation">Instalación</a></li>
      </ul>
    </li>
    <li><a href="#results">Resultados</a></li>
    <li><a href="#citation">Cita</a></li>
    <li><a href="#contact">Contacto</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## Proyecto

Tecnología y participación social para el mapeo de uso y cobertura del suelo en el Valle Inferior del Río Chubut

![satred](https://github.com/aletrujim/SatRed/blob/main/images/esquema_satred2.png)

El objetivo general de este trabajo es combinar procesos sociales y tecnológicos para desarrollar un mapa de UyCS del valle productivo, en base al algoritmo  desarrollado previamente *SatRed*, con una definición de categorías de  UyCS que resulte adecuada para la diversidad de usuarios/as del mismo. Respecto a los procesos tecnológicos, en este trabajo se utilizaron imágenes multi-temporales (combinación de imágenes de varias fechas), que permiten una mejor distinción/discriminación  entre los cultivos al capturar las diferencias que se producen en distintos momentos del año debido a diferentes ritmos de crecimiento, y estados fenológicos y fisiológicos de los cultivos. Respecto a los procesos sociales, se favorecieron diferentes espacios de intercambio de saberes y experiencias con personas de distintos ámbitos del valle mediante diferentes metodologías (talleres participativos, encuestas, encuentros, salidas a campo), que permitieron retroalimentar el proceso de elaboración del mapa de UyCS. 


<!-- GETTING STARTED -->
## Cómo empezar

Los modelos presentados en este trabajo se entrenaron en una máquina virtual patrocinada por Microsoft Azure 2, con el sistema operativo Windows Server 2016 Datacenter, de tamaño NC6 estándar (6 vCPUs [Intel Xeon CPU E5-2690 v3 2,60 GHz], con 56 GB de memoria RAM y un co-procesador GPU NVIDIA Tesla K80). Para incorporar el post-procesamiento se utilizaron los modelos pre-entrenados en la máquina virtual pero se corrieron en un equipo local con sistema operativo Fedora Linux 36 (Intel Core i7-4765T 2 Ghz, con 32 GB y un co-procesador GPU NVIDIA GeForce GTX 1660 Super). Los algoritmos se implementaron utilizando el lenguaje de programación Python.

### Requisitos previos

* GDAL
  ```sh
  pip install GDAL
  ```
* Tensorflow
  ```sh
  pip install --upgrade tensorflow
  ```
* scikit-learn
  ```sh
  pip install scikit-learn
  ```
  
 ### Instalación

1. Descargar imágenes de entrenamiento y validación [Link](https://drive.google.com/drive/folders/1HnXi9SyJOM9EH-nmxsAahzM1Z6Q94-i-?usp=sharing) o prepare las suyas propias. Si utiliza las imágenes descargadas, tenga en cuenta que se prepararon y particionaron con [Partition](https://github.com/aletrujim/SatRed/tree/main/partition)
2. Clonar el repositorio
   ```sh
   git clone https://github.com/aletrujim/SatRed.git
   ```
3. Instalar paquetes
   ```sh
   pip install -r requirements.txt
   ```
4. Ejecutar script python
   ```sh
   python satred.py --train=train --test=test --segmented=result --epochs=250
   ```
   
<!-- RESULTS -->
## Resultados

![satred results](https://github.com/aletrujim/SatRed/blob/main/images/mapa_satred2.png)
![satred results](https://github.com/aletrujim/SatRed/blob/main/images/results_satred2.png)

En Liberoff et. al (2024) el modelo SatRed v.2 se utilizó para elaborar un mapa de UyCS para el valle agrícola (48000 ha) de la cuenca baja del Río Chubut (VIRCh) para el año 2019 (Figura 1 - el mapa). El proceso de elaboración y validación del mapa se basó además en el uso de imágenes multi-temporales (combinación de información satelital de los meses de septiembre, octubre, diciembre del 2018 y enero, marzo y abril de 2019), en información de campo y en espacios de intercambio de saberes con actores sociales clave. Este mapa cuenta con 10 clases de UyCS y tiene una precisión global del 85%. La matriz de confusión (Tabla 1 - la matriz) muestra para cada intersección de filas y columnas tanto el número de píxeles (número central) como los valores normalizados por las filas y las columnas, expresados en porcentaje. Para el caso de la diagonal principal, estos porcentajes expresan la Precisión por clase (valores junto a las flechas con orientación vertical) y Exhaustividad por clase (valores junto a las flechas con orientación horizontal). Los valores fuera de la diagonal proveen información sobre la confusión entre clases.

<!-- CITATION -->
## Cita
Si utilizas estos datos o el modelo *SatRed v.2* en su investigación o trabajo, cite este trabajo
```
@article{trujillo2022satred,
  title={SatRed: New classification land use/land cover model based on multi-spectral satellite images and neural networks applied to a semiarid valley of Patagonia},
  author={Trujillo-Jim{\'e}nez, Magda Alexandra and Liberoff, Ana Laura and Pessacg, Natalia and Pacheco, Cristian and D{\'\i}az, Lucas and Flaherty, Silvia},
  journal={Remote Sensing Applications: Society and Environment},
  volume={26},
  pages={100703},
  year={2022},
  publisher={Elsevier}
}
```

<!-- CONTACT -->
## Contacto

Ana Liberoff - [liberoff@cenpat-conicet-gob.ar](liberoff@cenpat-conicet-gob.ar)

Project Link: [https://github.com/aletrujim/SatRed](https://github.com/aletrujim/SatRed)
 


