# sabanana

![app-icon](/docs/app-icon.png?raw=true "app-icon") 

Saba *(Musa balbisiana)* is a banana cultivar which are emerged and indigenous in the Philippines. It is considered as one of the core agricultural products in the country in terms of production and trade. The processed products derived from saba are continues to make a stronghold in both domestic and international markets. Also, it is widely useful in Filipino cuisine/deserts and in fact often used to extend or substitute staple food [[1]](http://cagayandeoro.da.gov.ph/wp-content/uploads/2013/04/SABA-BANANA-PRODUCTION-GUIDE.pdf). 

In a study [[2]](https://www.researchgate.net/publication/329268613_Effect_of_maturity_on_in_vitro_starch_digestibility_of_Saba_banana_Musa_%27saba%27_Musa_acuminata_x_Musa_balbisiana) (see Figure 1), mentioned that the Diamond Star Agro-Products Inc., Taguig City, Philippines have a commercial five different maturity stages for quality classification of saba — all green (stage 1), green but turning yellow (stage 2), greenish yellow (stage 3), yellow with green tips (stage 4), and yellow with brown flecks (stage 5). Owing the lack of instruments for an objective measurement of saba banana quality classification have to be done by visual inspection, which is time-consuming and costly hence, the purpose of this system.

![Saba five(5) stages](/docs/SabaFiveStages.PNG?raw=true "Saba five(5) stages") \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 1. Five (5) stages of Saba banana

### System Architecture & Process

![System Architecture](/docs/SystemArchitecture.png?raw=true "System Architecture") \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 2. System Architecture 

### Image Acquisition 
In a meanwhile a study [[7]](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/volume-25/issue-6/061410/Combining-multiple-features-for-color-texture-classification/10.1117/1.JEI.25.6.061410.short?SSO=1) further explained that the different viewpoint and illumination of the acquired digital image will greatly affect the classification accuracy of the system. So, in this case, Saba banana sample images must be acquired in a controlled environment using an objective instrument ― a customized photo box (see Figure 3 & 4 - more additional info [here](https://drive.google.com/drive/folders/1zj13Nhj801q7b42Yt_YwAsmAwzhU0X81?usp=sharing)) and a Samsung A7 camera in 1:1 ratio (4240x4240 pixels) with a distance of 18.5 cm above the sample and taken one pieces at a time as shown in a [video & images](https://drive.google.com/drive/folders/1hqCVoZ7U2_7YLUU3F9PUOgk7svjTfIt3?usp=sharing).
![PhotoBox Design Layout](docs\PhotoBoxDesign1.PNG?raw=true "PhotoBox Design Layout") \
![PhotoBox Design Layout](docs\PhotoBoxDesign2.PNG?raw=true "PhotoBox Design Layout") \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Figure 3 & 4. PhotoBox Design Layout 1 & 2 

### Testing Phase & Performance Evaluation
Tflite model in an android device with the testing dataset of 100 images (20 images per stage) was tested, the classifier achieved an 68% overall accuracy in correctly classifying the Saba banana sample images compared to the actual stage based on the classification made by the market vendor. Detailed information here in [excel file](https://drive.google.com/file/d/1CRg0NVjT6RguY_hnjwNerKUDtWSlAr0b/view?usp=sharing)

### Conclusion & Recommendation
The implemented system was created a tensorflow model in google colab & converted it into tflite model then tested in android device as a classifier and evaluated, it mostly fails to classify the Stage 2, 3 & 4. And it recommends in the future work, that a good TFLite model will be implemented with the data classify by an employee in Diamond Star Agro-Products Inc. & the images acquisition must be made the day during the Saba banana acquired.

### Install the debug APK and Test Saba images
1. Download the apk using the diawi link - https://i.diawi.com/tW548u (link expiration date: 11/30/2020) or just download the repo [app-debug.apk](https://github.com/nepmontanez/sabanana/blob/main/android/SabananaClassifier/app/build/outputs/apk/debug/app-debug.apk)
2. Access test images of sabanana [here](https://drive.google.com/drive/folders/1WTd0O0yl8tlv3eg4AXFwDrMlcFSIbvnT).

### Acknowledgements & Repo References:
* https://github.com/haruiz/AIPilipinas-Tflite
* [Flower_Classification_with_TFLite_Model_Maker.ipynb](https://colab.research.google.com/github/tensorflow/examples/blob/master/lite/codelabs/flower_classification/ml/Flower_Classification_with_TFLite_Model_Maker.ipynb#scrollTo=w-VDriAdsowu)
* [TensorFlow Lite Android image classification example](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/android)

### Softwares & Technologies used
* [Autocad 30 Day Free Trial](https://www.autodesk.com/products/autocad/free-trial?support=ADVANCED)
* [Google Colab](https://colab.research.google.com/github/tensorflow/examples/blob/master/courses/udacity_intro_to_tensorflow_for_deep_learning/l01c01_introduction_to_colab_and_python.ipynb)
* [Python Tensorflow](https://www.tensorflow.org/api_docs/python/tf)
* [Android Studio](https://www.googleadservices.com/pagead/aclk?sa=L&ai=DChcSEwiTqazNoKDtAhUSrJYKHRyOAHMYABAAGgJ0bA&ohost=www.google.com&cid=CAESQOD2AZ6OOar0hfMGS_2FivLqtWL7WDFZO0Ti6qpBWJJ8K5jRcK96pGs52tw_-4P7xTHlDyECzaEzVPA94-fOJHc&sig=AOD64_3_GS6dGOpThU7djb-CqZH8LWOxvw&q&adurl&ved=2ahUKEwiHkqbNoKDtAhViHKYKHWnPAj4Q0Qx6BAgaEAE)
* [Dia Diagram Editor](http://dia-installer.de/download/index.html.en)