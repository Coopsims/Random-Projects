# Spotify Popularity Prediction
## Author: Ben Funk
## CWID: 10866837
### Advanced Machine Learning

# Instructions for navigating Files and Opperating Code
## This Project was coded in Python version 3.11
### 1. Download Python 3.11 
   - The link can be found at: https://www.python.org/downloads/release/python-3119/
   - Download the correct version for your OS
   - Make sure with the first installation for python pops up, you click add to PATH. it should be centered at the bottom
   - #### PLEASE NOTE: I did all of my coding and testing on mac so you may have to install other dependancies outside of the ones listed, however I would not know what those are
### 2. Change Directory to the Project
   - From your terminal, follow these steps:
   - Specifically, you want to be in the file
   > Funk-Final-Submission
   - If you are on windows, there will be 2 folders in the main zip file, the funk-final one, and one that says MACOSX. This only appears on windows.
   - **DO NOT use the files in the MACOSX file**

### 3. Navigating the folders
   - There are four folders
     - In the **Final-Written-Report** You can find my demo video as well as the written report
     - In the **Datasets** folder are the raw .csv files which hold the spotify data
       - The file called *dataset.csv* holds about 100,000 rows of songs and is more evenly selected batch of data
       - The file called *tracks.csv* holds about 600,000 rows of songs and is less selected for an even balance of training data
     - In the **generated-images** folder, all of the output graphs are stored
       - This includes: 
       - 2 correlation matrices, one for each dataset
       - 4 confusion matrices, one for the model on the training data, and test data for each dataset
       - 1 average f1 score which looks at the average score across all popularity cutoffs and compares the two datasets. (Note: The SVM model was only trained on the smaller dataset as it takes O(n^3) time, so it having only one bar is not an error.)
       - 2 plots showing f1 score versus popularity cutoff
       - 1 KNN Kernel versus F1-Score for showing selected kernel value
       
### 4. Executable Python Files
   - Assuming that in your terminal you are now in the folder
   > Funk-Final-Submission
   - run the following command
   > pip install -r requirements.txt
   - This should load in every single package needed. if you get any error or warnings, don't worry about it; tensorflow is just unhappy
   - to train all of the different Ai models you can run
   > python ./Create-popularity-ml.py
   - If you are on windows and this doesn't work, reverse the forward slash to a backslash
   - This command will take between 1200 – 2000 seconds to fully execute depending on your CPU, it runs through 6 different machine learning models: Decision Tree, Random Forest, SVM, KNN Classifier, KNN Regressor, and a Neural Network.
   - Each model will print out to terminal its results, which will look like this:

   > scores for the training set:  
    Accuracy: 0.821619  
    Precision: 0.774078  
    Recall: 0.720153  
    F1 Score: 0.739428  
    AUPRC: 0.667985 <br><br>
    scores for the testing set:  
    Accuracy: 0.735731  
    Precision: 0.629849  
    Recall: 0.605106  
    F1 Score: 0.612552  
    AUPRC: 0.484475

   - These results have been saved to *.csv* files and are stored in the **Model-Results** folder
   - To Graph those results run the command
   > python ./model-comparison-graphing.py
   - This will print out the *.png* relating to F1-Score

