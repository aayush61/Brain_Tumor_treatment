This Folder contains the files related to Brain tumor dection. For this purpose first we need to train our model. 
You can train the model simply by running this script:<br>
<b>python Tumor_Detection_training.py</b><br><br>

After training your model you need to put MRI scan in a folder named as 'pred'. After that you simply run this code on your terminal:<br>
<b>python Tumor_Detection_prediction.py<b/><br><br>

This will return a Boolean value. 0 means the person does not have cancer and 1 means the person has a tumor.
