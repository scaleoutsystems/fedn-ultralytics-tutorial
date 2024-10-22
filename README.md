# FEDn Ultralytics Tutorial

## Introduction
This tutorial will guide you through the process of implementing a federated learning setup using the [FEDn] framework in combination with [Ultralytics] YOLOv8 models. Federated learning allows multiple clients to collaboratively train a global model without sharing their local datasets, ensuring data privacy and security. This tutorial is designed for users familiar with machine learning and federated learning concepts and will provide step-by-step instructions on configuring and running a complete federated learning workflow.

By the end of this tutorial, you will have built a distributed training environment where clients independently train local models, and a global model is aggregated on the server. You will also learn how to use Ultralytics YOLOv8 models for object detection tasks in this federated setting.

1.	Starting the server in FEDn – Learn how to initiate the central server that coordinates federated learning activities.
2.	Cloning the repository – Set up the project by cloning the necessary repository for configuration and deployment.
3.	Installing prerequisites – Install all required dependencies for the client environments.
4.	Setting up the dataset – Properly structure and configure your dataset to be used with Ultralytics models.
5.	Setting up configurations – Configure the model, such as defining the number of classes for the YOLOv8 model.
6.	Building the compute package – Build the compute package to prepare for the training process.
7.	Initializing the seed model – Generate the initial model to start the training process.
8.	Initializing the server-side – Set up the server-side of the federated learning system.
9.	Connecting and starting the clients – Connect the clients that will participate in the federated training.
10.	Training the global model – Observe how the global model improves through the aggregation of client updates and monitor training progress.

By following these steps, you will not only gain hands-on experience with the FEDn platform but also learn how to integrate object detection tasks with YOLOv8 in a federated learning environment.

## Prerequisites
-  `Python >=3.8, <=3.12` <https://www.python.org/downloads>

## Step 1: Starting the server in FEDn
Firstly, create an account on the FEDn studio platform <https://fedn.scaleoutsystems.com/signup>.
Once you are logged in, you need to start a new project by clicking on the `New project` button.
This initializes the server which later will be used to run the federated learning process.

## Step 2: Cloning the repository
Next, you need to clone the repository:
```bash
git clone https://github.com/scaleoutsystems/ultralytics-implementation-in-fedn-tutorial
```
Then navigate into the repository:
```bash
cd ultralytics-implementation-in-fedn-tutorial
```
This repository contains all the necessary files and configurations for the federated learning setup.

## Step 3: Installing prerequisites
Next, you need to install the prerequisites. You can install everything using pip:
```bash
pip3 install -r requirements.txt
```
This is recommended to be done in a virtual environment to avoid conflicts with other packages.

## Step 4: Setting up the dataset
Ultralytics stores all datasets in a specific directory. You can set the location of this directory by configuring the datasets_dir option with yolo settings. To do this, run the following command:
```bash
yolo settings datasets_dir=<path_to_dataset>
```
Replace <path_to_dataset> with the path to your datasets folder, which in this tutorial is named 'datasets'. Then your data needs to be placed within a folder called 'fed_dataset'. The structure should be as follows:
```bash
datasets/
  fed_dataset/
    train/
      images/
        image1.jpg
        image2.jpg
        ...
      labels/
        image1.txt
        image2.txt
        ...
    val/
      images/
        image1.jpg
        image2.jpg
        ...
      labels/
        image1.txt
        image2.txt
        ...
```

Each label file should correspond to an image file, and the format of each label should be:

```bash
<class> <x_center> <y_center> <width> <height>
<class> <x_center> <y_center> <width> <height>
...
```
Each line corresponds to one bounding box in the image.

For further details on how to prepare your dataset, you can visit <https://docs.ultralytics.com/datasets/>.

For getting started quickly with a sample dataset, you can navigate into the `examples` repository to download and partition a sample dataset.

## Step 5: Setting up configurations

### Number of classes
To set up your Ultralytics model, you need to adjust the configuration files. Specifically, the number of classes (nc) must be set in both the `data.yaml` and the `yolov8_.yaml` files. Make sure to update these files with the appropriate number of classes for your specific dataset.

### Size of the model
You also need to select which YOLOv8 model to use by renaming the `yolov8_.yaml` file according to the desired model variant:
- For YOLOv8n (nano), rename the file to `yolov8n.yaml`
- For YOLOv8s (small), rename the file to `yolov8s.yaml`
- For YOLOv8m (medium), rename the file to `yolov8m.yaml`
- For YOLOv8l (large), rename the file to `yolov8l.yaml`
- For YOLOv8x (extra large), rename the file to `yolov8x.yaml`

### Local client configurations
Each client can set different training configurations in the `client_config.yaml` file. This file contains the configurations for the client environments, such as the number of local epochs, and batch size. You can adjust these configurations to suit each client's hardware and training requirements.

##  Step 6: Building the compute package
Once you’ve completed all the configurations, you can build the compute package by running the following command:
```bash
fedn package create -p client
```
The compute package contains all the necessary files and configurations for the client environments.
If you make any changes to the configurations later, you’ll need to rebuild and reupload the compute package to apply the updates.

## Step 7: Initializing the seed model
To initialize the seed model, run the following command:
```bash
fedn run build -p client
```
This command will generate the seed model that will be used as the starting point for the federated learning process.

## Step 8: Initializing the server-side
The next step is to initialize the server side. This is done by uploading the compute package and the seed model to the FEDn studio platform.
This is done by pressing the Sessions button and then the "New session" button. Here you can upload the compute package `package.tgz` and the seed model `seed.npz`.
Once the compute package and seed model are uploaded, you can create the session by pressing the "Create session" button. Here you also configure the total number of rounds and aggregator function for the federated learning process.

Before starting the training process, you need to connect the clients to the server which is done in the next step.

## Step 9: Connecting and starting the clients
To connect a client to the server, you need to hand each client a client.yaml file. This file contains the necessary configurations for the client to connect to the server and gain access to the compute package. Connect clients by pressing the "Clients" button on the left hand side. Here you can download a client.yaml file for each client.
Place the client.yaml file in the repository and start the client by running following command:
```bash
fedn client start -in client.yaml --secure=True --force-ssl
```
This starts the client and connects it to the server. Repeat this process for each client you want to connect to the server.


## Step 10: Training the global model
Once the clients are running, you can start the global training by pressing the Start session button in FEDn studio. This will initiate the federated learning process, where the global model is trained by aggregating the updates from the clients. Now you will see things happening on both the server and client side. You can monitor the training progress on the FEDn studio platform where metrics such as recall, precision, and mAP scores are shown.

<img src="figs/global_convergence.png" width=80% height=80%>


Once training is completed, you can download any model from a certain round in the session. The model can be used for inference on new data.

## Conclusion
In this tutorial, you have learned how to implement Ultralytics YOLOv8 models in a federated learning setting using the FEDn platform. By following the steps outlined in this tutorial, you have successfully set up a distributed training environment where clients independently train local models, and a global model is aggregated on the server. You have also learned how to configure the dataset, set up the model configurations, build the compute package, and start the federated learning process. By completing this tutorial, you have gained hands-on experience with federated learning and object detection tasks using Ultralytics models.

# Note
Steps 1, 2, 5, 6, 7 and 8 only need to be done once to set up the federated learning environment in FEDn.

To connect a new client, the only steps that needs to be followed are step 3, 4 and 9.
Each client can have different configurations in the client_config.yaml file. This file needs to be downloaded from the repository separately for each client.