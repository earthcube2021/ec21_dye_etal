## Toward autonomous detection of anomalous GNSS data via applied unsupervised artificial intelligence 

## Authors
Mike Dye, D. Sarah Stamps, Myles Mason

## Abstract
Artificial intelligence applications within the geosciences are becoming increasingly common, yet there are still many challenges to adapt established techniques to geoscience datasets. Applications in the realm of volcanic hazards assessment show great promise. This Jupyter notebook ingests real-time GNSS data streams from the EarthCube CHORDS (Cloud-Hosted Real-time Data Services for the Geosciences) portal TZVOLCANO, applies unsupervised learning algorithms to perform automated data quality control (“anomaly detection”), and explores autonomous detection of unusual volcanic activity. TZVOLCANO streams real-time GNSS positioning data from the active Ol Doinyo Lengai volcano in Tanzania through UNAVCO’s real-time GNSS data services, which provide near-real-time positions processed by the Trimble Pivot system. The positioning data (latitude, longitude, and height) are imported into this Jupyter Notebook in user-defined time spans. The raw data are then collected in sets by the notebook and processed to extract useful calculated variables, such as an average change vector, in preparation for the machine learning algorithms. In order to create usable data, missing time-series points are imputed using the “most frequent” strategy. After this initial preparation, unsupervised K-means and Gaussian Mixture machine learning algorithms are utilized to  locate and remove data points that are likely unrelated to volcanic signals. Preliminary results indicate that both the K-means and Gaussian Mixture machine learning algorithms perform well at identifying regions of high noise within tested GNSS data sets.

## Keywords
TZVOLCANO, CHORDS, UNAVCO, Artificial Intelligence, Machine Learning

## Binder
These notebooks are Binder enabled and can be run on [mybinder.org](https://mybinder.org/) for free using the links below

### MD_01_TZVOLCANO_Gaussian_Mixtures_Anomaly_Detection (Notebook for EarthCube) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mdye/TZVOLCANO_machine_learning.git/HEAD?filepath=MD_01_TZVOLCANO_Unsupervised_AI_Anomaly_Detection.ipynb)


## Installation on local machine

It is a best practice to use a virutal python environment so the differing dependencies between projects do not conflict


To run this Notebook on your local machine, you will need the following Python3 libraries:
| Library  | Version | |
| ------------- | ------------- | ------------- |
| numpy | 1.19.* | Linear algerbra and array manipulation |
| pandas | 1.2.* | Data analysis and manipulation |
| scipy | 1.6.* | Science and math utilities |
| matplotlib | 3.3.* | Plotting |
| ipywidgets | 7.6.0 | GUI widgets |
| scikit-learn | 0.24.* | Machine learning  |
| tensorflow | 2.5.0rc3 | Neural network libraries |



### Installing on a local environment using conda
If you are using conda the collowing code will create the required environment for this notebook
```
conda create --name tzvolcano_machine_learning_env
conda activate tzvolcano_machine_learning_env
conda install python=3.7
conda install pip
conda install -c conda-forge jupyterlab
conda install -c conda-forge notebook
conda install numpy=1.19
conda install pandas=1.2
conda install scipy=1.6
conda install matplotlib=3.3
conda install ipywidgets=7.6
conda install scikit-learn=0.24
pip install tensorflow==2.5.0.rc3
```


Note that the installation of tensorflow if done with pip, which may cause problems when used in conjustion with conda.
Unfortunately, the version of tensorflow required has not been released as a conda package as of this writing.



### Installing dependencies with pip3
If you are not using conda, you can install the depenencies with pip3

All the required dependencies can be installed from the requirements.txt file:

`pip3 install -r requirements.txt`

#### Create a new environment
`python3 -m venv tzvolcano_ml_env`

#### Activate the environment
`source env/bin/activate`

when done with the notebook, you can deactivate the environment:

`deactivate`

More information on virtual envirnments can be found [here](https://realpython.com/python-virtual-environments-a-primer/)



### Enable the table of contents extension
```
pip install jupyter_contrib_nbextensions
jupyter contrib nbextension install 
jupyter nbextension enable toc2/main
```





