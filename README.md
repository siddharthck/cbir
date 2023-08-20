# cbir
### Content based image retrieval - tensorflow, VGG-16, python3, keras 

## Technology stack

### Python : 3.7

### Web framework : Flask

### Dependencies : 
see [tensorflow.yml](tensorflow.yml) file.

### Dataset used : 
caltech256 (please download from oficial website:http://www.vision.caltech.edu/Image_Datasets/Caltech101/)

## How to run :

### Step 1 : Create and store image features in features.h5 file
open CBIR2.pynb file in jupyter notebook. (make sure u are using python which has tensorflow installed)

Follow instructions given in pynb file itself. 
features.h5 file we created will be used in the actual flask app !



### Step 2 : Run the app

(activate virtual env if you have used)

change directory to flaskCBIR
```
$ cd flaskCBIR

$ python3 fun.py
````

## Results 
### Home page !
![Alt text](/screenshots/sc1.png?raw=true&sanitize=true "Home page")

### Browse and select image to search ! 

![Alt text](/screenshots/sc2.png?raw=true&sanitize=true "Home page")

### Final result

![Alt text](/screenshots/sc3.png?raw=true&sanitize=true "Home page")

### Let's do one more search

![Alt text](/screenshots/sc4.png?raw=true&sanitize=true "Home page")





## Happy coding !
 



