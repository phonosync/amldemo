# amldemo
First steps in using the Azure Machine Learning Service

For convinience some utility functions are organised in a separate repository which are included here as a submodule.

* Image Classification Model based on MNIST dataset  
Demonstrates how you can execute a locally developed script that on a remote compute ressource.
Based on the official tutorial https://docs.microsoft.com/en-us/azure/machine-learning/service/tutorial-train-models-with-aml
    * https://github.com/phonosync/amldemo/blob/master/mnist_local.ipynb
    * https://github.com/phonosync/amldemo/blob/master/mnist_remote.ipynb

# Setup and Requirements
* Make sure to include the submodule with  
git clone --recursive https://github.com/phonosync/amldemo.git
* Developed with Python 3.7
* Jupyter Notebook (some of the azure ml widgets do not work/are not available in Jupyter Lab)
* Build your Python environment based on the requirements.txt file in the project root
* Create a new Machine Learning Service Workspace in portal.azure.com (or use a shared one)
* Download the configuration ('download config.json' from overview pane)

