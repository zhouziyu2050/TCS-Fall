# TCS-Fall
Falls pose a serious health risk for the elderly, particular for those who are living alone. The utilization of WiFi-based fall detection, employing Channel State Information (CSI), emerges as a promising solution due to its non-intrusive nature and privacy preservation. Despite these advantages, the challenge lies in optimizing cross-individual performance for CSI-based methods. 

This study aimed to develop a resilient real-time fall detection system across individuals utilizing CSI, named TCS-Fall. This method was designed to offer continuous monitoring of activities over an extended timeframe, ensuring accurate and prompt detection of falls. 

Extensive CSI data on 1800 falls and 2400 daily activities was collected from 20 volunteers using CSI tool. The grouped coefficient of variation (GCV) of CSI amplitudes were utilized as input features. These features capture signal fluctuations and are input to a convolutional neural network (CNN) classifier. Cross-individual performance was extensively evaluated using various train/test participant splits. Additionally, a user-friendly CSI data collection and detection tool was developed using PyQT. To achieve real-time performance, data parsing and preprocessing computations were optimized using Numba's just-in-time compilation.

The proposed TCS-Fall method achieved excellent performance in cross-individual fall detection. On the test set, AUC reached 0.999, no error warning ratio (NEWR) score reached 0. 955 and correct warning ratio (CWR) score reached of 0.975 when trained with data from any 2 volunteers. Performance can be further improved to 1.00 when 10 volunteers were included in training data. The optimized data parsing/preprocessing achieved over 20x speedup compared to previous method. The PyQT tool parsed and detected the fall within 100ms.

TCS-Fall method enables excellent real-time cross-individual fall detection utilizing WiFi CSI, promising swift alerts and timely assistance to elderly. Additionally, the optimized data processing led to a significant speedup. These results highlight the potential of our approach in enhancing real-time fall detection systems.

# Keywords
Channel state information, cross-individual fall detection, grouped coefficient of variation, real-time detection, time-continuous stack sample

# download dataset
All raw data (".dat" file) can be downloaded at：
* https://pan.baidu.com/s/16mKF9zF-mKx0H2ojYJNNJA?pwd=nxpi
* https://ieee-dataport.org/documents/tcs-fall

# Run Code
* Firstly, change the params and run ```preprocess.ipynb``` in jupyter lab.
* Secondly, run ```butter-cv-cnn.py``` and the result will be record in ```result.csv```.
* Finally, run ```result_analyse.ipynb``` to analyse the result and get the bar images.

# CSI Data Visualization
Run ```csi-view.ipynb``` to show the CSI data.

# Package dependencies
CSITool can be accessed from https://github.com/dhalperi/linux-80211n-csitool-supplementary.

Python 3.8+ is required, and the main dependency packages are as follows:
```
pip install jupyter
pip install torch==1.11.0 torchvision==0.12.0
pip install numpy==1.23.5 numba==0.56.4 PyWavelets=1.4.1 scipy==1.9.3 matplotlib==3.6.2 pandas==1.5.2
pip install onnxruntime
```

# Cite
```
@article{zhou2024tcs,
  title={TCS-Fall: Cross-individual fall detection system based on channel state information and time-continuous stack method},
  author={Zhou, Ziyu and Liu, Zhaoqing and Liu, Yujie and Zhao, Yan and Wang, Jiarui and Zhang, Bowen and Xia, Youbing and Zhang, Xiao and Li, Shuyan},
  journal={Digital Health},
  volume={10},
  pages={20552076241259047},
  year={2024},
  publisher={SAGE Publications Sage UK: London, England}
}
```
