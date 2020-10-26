🌊Coastal sea elements prediction with U-Net based models
========

Official code from the paper that you can find in the following link:

📊 Results
-----

Some animation of the actual vs prediction of the AsymmInceptionRes-3DDR-UNet model, using a 48h ahead prediction:

+-----------------------------+----------------------------------------+--------------------------------------------+
|       Variable              | Actual                                 | Prediction                                 |
+=============================+========================================+============================================+
| Sea Surface Height          |.. figure:: figures/actual-SSH.gif      |.. figure:: figures/prediction-SSH.gif      |                                                             
+-----------------------------+----------------------------------------+--------------------------------------------+
| Sea Water Salinity          |.. figure:: figures/actual-SAL.gif      |.. figure:: figures/prediction-SAL.gif      |                      
+-----------------------------+----------------------------------------+--------------------------------------------+
| Eastward Current Velocity   |.. figure:: figures/actual-CUR-uo.gif   |.. figure:: figures/prediction-CUR-uo.gif   |
+-----------------------------+----------------------------------------+--------------------------------------------+
| Northward Current Velocity  |.. figure:: figures/actual-CUR-vo.gif   |.. figure:: figures/prediction-CUR-vo.gif   |                                                             
+-----------------------------+----------------------------------------+--------------------------------------------+

Note: the dark area is made of pixels that correspond to the land.


💻 Installation
-----

The required modules can be installed  via:

.. code:: bash

    pip install -r requirements.txt
    
Quick Start
~~~~~~~~~~~
To launch the training, please run:

.. code:: bash

    python train_selected_model.py 

📜 Scripts
-----

- The scripts contain the models, the data preprocessing, as well as the training files.

🔍 Models
-----

We show here the schema related to the AsymmInceptionRes-3DDR-UNet model.

.. figure:: figures/AsymmInceptionRes-3DDR-UNet.png
  
📂 Data
-----

In order to download the data, please email to one of the following addresses:

siamak.mehrkanoon@maastrichtuniversity.nl

j.garciafernandez@student.maastrichtuniversity.nl

i.alaouiabdellaoui@student.maastrichtuniversity.nl

The data must be downloaded and unzipped inside the 'Data/' directory.


🔗 Citation
-----

If you decide to cite our project in your paper or use our data, please use the following bibtex reference:

.. code:: bibtex

    @article{Fernández2020coastal,
        title={Coastal sea elements prediction using U-Net based models},
        author={García Fernández, Jesús and Alaoui Abdellaoui, Ismail and Mehrkanoon, Siamak},
        journal={arXiv preprint arXiv:},
        year={2020}
    }
