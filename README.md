# TCS-Fall
Falling is considered as a serious problem for the aging society, especially for the elderly who is living alone. In recent years, the recognition accuracy of human activities, such as falling down, based on channel state information (CSI) has been gradually improved. However, the problems of cross-domain (person or position, etc.) are still bothering researchers with low recognition accuracy. Here in this work, a new cross-person fall detection method, called TCS-Fall, was proposed. Unlike other works, this method focused more on changes in CSI amplitude caused by human activities, rather than the amplitude itself. To verify the advantages of this method, 20 volunteers were recruited, each with 7 group of single-action (SA) samples and 6 group of continuous time stack samples. The Butterworth low-pass filter is utilized here to filter ambient noise greater than 150Hz, and the matrix of the coefficient of variation of the amplitude was calculated. Then the matrix is trained by CNN algorithm. In order to avoid data contamination, the k-fold cross-validation method was used to divide the data into training datasets and validation datasets according to volunteer IDs. To achieve real-time fall detection, numba's precompiled technology is used in data preprocessing, and Open Neural Network Exchange (ONNX) models were used to run deep learning models. As a result, no error warning ratio (NEWR) within 150 seconds reached 95.46% when the data of only 2 volunteers were used for training. And NEWR can reach 100% when that data of 10 volunteers is used for training.

# download dataset
All raw data (".dat" file) can be downloaded at：
* https://pan.baidu.com/s/1NKy0UEjHSAgaA8AHq6iHYw?pwd=4y1d

# Run Code
* Firstly, change the params and run ```preprocess.ipynb``` in jupyter lab.
* Secondly, run ```butter-cv-cnn.py``` and the result will be record in ```result.csv```.
* Finally, run ```result_analyse.ipynb``` to analyse the result and get the bar images.

# CSI Data Visualization
Run ```csi-view.ipynb``` to show the CSI data.

# Package dependencies
CSITool can be accessed from https://github.com/dhalperi/linux-80211n-csitool-supplementary.

Python 3.8+ is required, and the main dependency packages are the follows.
```
pip install jupyter
pip install torch==1.11.0 torchvision==0.12.0
pip install numpy==1.23.5 numba==0.56.4 PyWavelets=1.4.1 scipy==1.9.3 matplotlib==3.6.2 pandas==1.5.2
pip install onnx onnxoptimizer onnxruntime-gpu protobuf
```
