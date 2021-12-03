<div id="top"></div>





<!-- PROJECT LOGO -->
<br />



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

Glass fragments are one of the most frequently used items in forensic science.
In most of the crime scenes such as house-breaking, even small fragments of the glass attached to the clothes of a suspect would solve the problem.
Using elemental composition and refractive index, we can tell where the glass comes from, even from small glass fragments.
Is it from a house window? car windshield? Bottle? 
Making a robust ML model with a good size of data will provide a good tool to solve such problems.

<p align="right">(<a href="#top">back to top</a>)</p>



### Built With


* [Scikit-learn](https://scikit-learn.org/)
* [tensorflow](https://www.tensorflow.org/)
* [imblearn](https://imbalanced-learn.org/)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Here's how to get started with the project.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* sklearn
  ```sh
  pip install sklearn
  ```
* Tensorflow
  ```sh
  pip install tensorflow
  ```  
* imblearn
  ```sh
  pip install imblearn
  ```  
* pandas
  ```sh
  pip install pandas
  ```    
* numpy
  ```sh
  pip install numpy
  ```    
* matplotlib
  ```sh
  pip install matplotlib
  ```   

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/senay-dev/Glass-Identification.git
   ```
2. Folder Structure: --<br />
                      |----Model Building<br />
                      |--------Project_script.ipynb<br />
                      |--------Project_script.py<br />
                      |----Make Predictions<br />
                      |--------run.py<br />
                      |--------ANN.h5<br />
                      |--------...(other saved models)<br />
                      |----Presentation.pptx<br />
                      |----README.md<br />
3. On Model Building Folder, you can run the same code that was used to build all models with techniques such as SMOTE oversampling and hyperparameter tuning.
   To run project_script.py:
        ```sh
         python Project_script.py
        ```
   NOTE: runing Project_script.py will take time, to view the results right away (including the plots), please open Project_script.ipynb.
4. On Make Predictions folder, you can make predictions on one sample by giving it the single data or you can make it make predictions on a portion of the file called glass.data. Details are <a href="#usage">here</a>.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

In Make Predictions Folder,

* To make predictions:
```sh
   python run.py
```
It will ask you an input, and you can provide one. Try this: 1.52211,14.19,3.78,0.91,71.36,0.23,9.14,0.00,0.37

* To make prediction on portion of glass dataset
```sh
   python run.py 
```
It will ask you for an input, type in `f [portion_size]`. for example `f 0.3`. And the code will sample data and provide accuracy score.

<p align="right">(<a href="#top">back to top</a>)</p>




<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/senayfre45) - senayfre0@gmail.com

Project Link: [https://github.com/senay-dev/Glass-Identification](https://github.com/senay-dev/Glass-Identification)

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments
Dataset: https://archive.ics.uci.edu/ml/datasets/Glass+Identification <br />
Logistic Regression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html <br />
SVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html <br />
KNN: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html <br />
Decision Tree Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html <br />
Random Forest Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html <br />
https://machinelearningmastery.com/smote-oversampling-for-imbalanced-classification/ <br />
https://amirhessam88.github.io/glass-identification/ <br />
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge
[contributors-url]: https://github.com/othneildrew/Best-README-Template/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/othneildrew/Best-README-Template.svg?style=for-the-badge
[forks-url]: https://github.com/othneildrew/Best-README-Template/network/members
[stars-shield]: https://img.shields.io/github/stars/othneildrew/Best-README-Template.svg?style=for-the-badge
[stars-url]: https://github.com/othneildrew/Best-README-Template/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=for-the-badge
[issues-url]: https://github.com/othneildrew/Best-README-Template/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
