
   **Note: If you are new to FEDn, we recommend that you start with the MNIST-Pytorch example instead: https://github.com/scaleoutsystems/fedn/examples/mnist-pytorch**

# African Wildlife Example

This is an example FEDn project that trains the Ultralytics YOLOv8n model to classify buffalos, elephants, rhinos and zebras. See a few examples below,

<img src="figs/buffalo.jpg" width=30% height=30%>

<img src="figs/elephant.jpg" width=30% height=30%>

<img src="figs/rhino.jpg" width=30% height=30%>

<img src="figs/zebra.jpg" width=30% height=30%>



## How to run the example

To run the example, follow the steps below. For a more detailed explanation, follow the Quickstart Tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html



### 1. Prerequisites

-  `Python >=3.8, <=3.12` <https://www.python.org/downloads>
-  `A project in FEDn Studio`  <https://fedn.scaleoutsystems.com/signup>


### 2. Install FEDn and clone GitHub repo

Install fedn: 

``` 
pip install fedn
```

Clone this repository, then locate into this directory:

```
git clone https://github.com/scaleoutsystems/fedn-ultralytics-tutorial.git
cd fedn-ultralytics-tutorial/examples/african-wildlife
```

### 3. The dataset

1. Create a directory inside the african-wildlife folder
```
mkdir datasets
```
2. Download the dataset from https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip
3. Unzip and move the folder to the "datasets" directory you created in step 1 

Then run the script partition_data.py to split the dataset into random partitions to distribute to the clients.

```bash
python3 partition_data.py <number of partitions>
```

Note: Each client needs to call their dataset partition the same, so rename the datasets after distributing them.

### 4. Creating the compute package and seed model

Create the compute package:

```
fedn package create --path client
```

This creates a file 'package.tgz' in the project folder.

Next, generate the seed model:

```
fedn run build --path client
```

This will create a model file 'seed.npz' in the root of the project. This step will take a few minutes, depending on hardware and internet connection (builds a virtualenv).  

### 5. Running the project on FEDn

To learn how to set up your FEDn Studio project and connect clients, take the quickstart tutorial: https://fedn.readthedocs.io/en/stable/quickstart.html. When activating the first client, the dataset inside your datasets directory and the data.yaml file will be moved to dedicated client folders to mimic the behavior of a distributed setup where the data sits locally at the client.   

