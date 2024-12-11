# `480_Semester_Project`

## CNN

### (AI vs Real Detection)

## [`Nathan’s 480_Semester_Project Repo`](https://github.com/nhasey/480_Semester_Project)  
To use our model, open FinalCNN.py
Make sure to use a conda environment.
Install torch and torchvision.  
You may need to use the commented out line 17 & 18 to get the path on your computer when copy pasting it make sure to add /real-vs-fake in the end. It will not be there automatically and will cause you issues if it is not added. This step will ensure that the database has been installed on your computer and is accessible.  
After making the necessary adjustments to the path, Kaggle credentials, making sure to adjust the username part of the path to your username instead of oussamanouadir as in FinalCNN.py.  
Finally, run FinalCNN.py, which will save a pre-trained model to the file path you are executing this from.

## Web application, flask API, and pre-trained model.

### (reality|check)

## [`Vas’s reality-check Repo`](https://github.com/vas2000-emu/reality-check)  
If you want to create the web application yourself, the following would be necessary:  
[Download and install Node.js](https://nodejs.org)  
[Facebook’s create-react-app repo](https://github.com/facebook/create-react-app?tab=readme-ov-file)  
[Online Guide to Creating a react app](https://create-react-app.dev/)  
Now, skipping ahead, you would have to deploy this by connecting your web application project to a git repo, then creating an account on netlify.com to launch the web application using that git repo.

### [`reality|check`](https://realitycheck480.netlify.app/)

To use this already deployed web application, click on the link above to take you to it. Once there, click the BEGIN HERE button. This will take you to a Google Drive folder that contains a README.txt file. In this file you will find the instructions on how to install the requirements listed in requirements.txt, file location instructions for the files cnn_weights.pth and app.py, as well as how to use the web application to detect whether a supplied human face image file is real or AI-generated, with the percentage probability of that determination.
