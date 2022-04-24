# HOOF-based-MEdetection
A python program to detect micro expression(ME) in video with HOOF descriptors and SVM classifier. 

<img width="200" alt="faceDetected landmarks" src="https://user-images.githubusercontent.com/22282000/164990572-965e3ef8-48ef-48e3-8915-1e87c27979e2.png">

--------------------------------------------------

Il programma si occupa della detection, tramite istogramma del flusso ottico e classificatore SVM, di micro espressioni facciali involontarie presenti nel dataset CASME2. <br />
Per maggiori dettagli vedere il file pdf della documentazione e il file con le istruzioni per l'esecuzione. <br />
Il programma utilizza varie librerie di image processing e machine learning tra cui opencv, dlib, seaborn, pandas, sklearn e per la classificazione fa uso di libSVM. <br />

-------------------------------------------------

# Links utili e riferimenti 
- [LibSVM site](https://www.csie.ntu.edu.tw/~cjlin/libsvm/)
- [OpenCV optical flow](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html)
- [CASME2 dataset](http://fu.psych.ac.cn/CASME/casme2-en.php)
- [Shape predictor landmarks](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat)
- [article1](https://arxiv.org/abs/1903.10765)
- [article2](https://arxiv.org/abs/1812.10306)
- [article3](https://www.researchgate.net/publication/319769812_Reading_Hidden_Emotions_Spontaneous_Micro-expression_Spotting_and_Recognition)
