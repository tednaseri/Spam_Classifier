# Spam_Classifier
<p align="justify">Nowadays, spam detection is one of the major applications of Machine Learning. As an example, email service providers use this technique to detect spam email (Junk mail) and prevent them to be accessed in male box. Like email services, we can use spam detection technique for analyzing short message service (SMS) as well. Before getting into the project, let us first define what is a spam message. They are unwanted or unexpected messages that shown up in user cell phone. Normally, they are not only annoying, but also can be dangerous too. For instance, they can contain some links that lead to phishing web sites or similarly websites that are hosting malware. Since the spam messages are sent from unknown phone numbers, they are trying to be more eye-catching than usual messages. For example, it is likely to have words such as win, chance, prize and so on. To detect spam messages, we can take advantage of supervised machine learning algorithms if we have enough labeled dataset. <br>
In this project, I have designed and implemented a machine learning model that can efficiently predict if a text message is spam or not. Finally, this project is also implemented in form of a <strong>machine learning web application.</strong></p>

# Files and Folders in this Repository:
### Machine_Learning Folder:
There are files inside this folder:
1) Utils_Spam_Classifier: This file includes common python functions which will be called insider other two files.
2) Preprocess_EDA_Spam_Classifier: This files will do preprocess and explanatory data analysis on the dataset. Then it will produce a modified version of dataset.
3) Modeling_Spam_Classifier_Spam_Classifier: This file will apply machine learning modeling on the dataset.

### Web Folder:
The content of this folder are for machine learning web application. Inside this folder we have one file and two other folders. 
1) Flask_Web_Application: This file includes python codes for web application. 
2) Static/css/style.ss: This file includes css file for the interface of the website.
3) templates: Inside templates, there are 4 HTML files for the web pages.

### Model Folder:
The optimized model is saved in this folder.

### Data Folder:
There are three datasets in this folder:
1) SMSSpamCollection: It is the original dataset that can be download from UCI repository.
2) df_new: It is one of the modified version of the dataset that I made.
3) df_source: It is one of the modified version of the dataset that I made.

# More Details:
<p>More details is provided in my personal website at http://tednaseri.pythonanywhere.com/spam_classifier</p>
