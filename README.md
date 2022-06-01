## SatRed: New classification land use/land cover model based on multi-spectral satellite images and neural networks applied to a semiarid valley of Patagonia

![satred](https://github.com/aletrujim/SatRed/blob/main/images/Fig3.png)

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
    <li><a href="#citation">Citation</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

![satred](https://github.com/aletrujim/SatRed/blob/main/images/Fig1.png)

In this article we describe a new model, SatRed, which classifies land use and land cover (LULC) from Sentinel-2 imagery and data acquired in the field. SatRed performs pixel-level classification and is based on a densely-connected neural network. The study site is the lower Chubut river valley which has an extension of 225 km2 and is located in estern semiarid Patagonia. SatRed showed a 0.909 ± 0.009% (mean ± sd, n = 7) overall accuracy and outperformed the seven most traditional Machine Learning methods, including Random Forest. Our model accurately predicted buildings, shrublands, pastures and water and yielded the best results with classes harder to classify by all methods considered (Fruit crops and Horticulture). Further improvements involving textural information and multi-temporal images are proposed. Our model proved to be easy to run and use, fast to execute and flexible. We highlight the capacity of SatRed to classify LULC in small study areas as compared to large data sets usually needed for state-of-the-art Deep Learning models suggested in literature.

<!-- GETTING STARTED -->
## Getting Started

All procedures were computed in a virtual machine sponsored by Microsoft Azure 2, with the operating system Windows Server 2016 Datacenter, of standard NC6 size (6 vCPUs [Intel Xeon CPU E5-2690 v3 2.60 GHz], 56 GB RAM memory and a NVIDIA Tesla K80 GPU coprocessor). Algorithms were implemented using the Python programming language. The Rasterio package (Gillies et al., 2013) was used to access and process geospatial raster data, and Shapely (Gillies et al., 2019) for manipulation of polygons.

To run a local copy of this project follow these simple steps:

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

1. Download training and validation images [Link](https://drive.google.com/drive/folders/1HnXi9SyJOM9EH-nmxsAahzM1Z6Q94-i-?usp=sharing) or prepare your own. If you use the downloaded images, please note that they were prepared and partitioned with [Partition](https://github.com/aletrujim/SatRed/tree/main/partition)
2. Clone the repo
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

![satred results](https://github.com/aletrujim/SatRed/blob/main/images/Fig2.png)

We developed a Neural Networks-based model for pixel-based classification of satellite imagery to map land use and land cover, including crops. SatRed has a relatively low requirement of data compared to state-of-the-art deep learning models suggested by the literature. In addition, SatRed has an overall good performance and outperforms other seven machine learning methods by at least 3.5% in overall precision scores. It also stands out for being more stable in individual classes, showing better scores for all classes being analysed. SatRed training is also faster than Nearest Neighbors which also shows acceptable scores.

<!-- CITATION -->
## Citation
If you use this data or the *SatRed* model in your research, please cite this project.
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
## Contact

Alexa Trujillo - [@aletrujim](https://twitter.com/aletrujim)

Project Link: [https://github.com/aletrujim/SatRed](https://github.com/aletrujim/SatRed)
 

