# Capacity-Optimization

The implementation is divided into the following steps:
Step 1 to Step 9 are for a large network. After being fully trained, the large network is used to teach a small network.
1. Train a large SM to segment out all person parts as one category. This is much simpler than 7 class prediction and the network can roughly segment out all people.
2. Also use LM to predict the skeletons of all people.
3. Based on the results from Step 1 and Step 2, we can roughly segment out the regions containing people and set the backgrounds to zeros. This can reduce the workload of CNNs. 
4. Enlarge the region if interested regions are too small and re-use LM to generate skeleton on the enlarged regions.
5. Train and test on the simplified images.
6. Based on the similarity of skeletons, choose some people from training images to simulate those from test images. Especially simulate the cases where people in test images are small, or of hard poses.
7. Data augmentation: Scaling, Rotation, Color(Add a value to the hue channel of the HSV representation) Also we need to random the backgrounds and foregrounds.
8. Re-proposed the SM for colorization, train it on the sketched version of test images and use it to colorize the training images to enrich the training data. 
9. Use several networks to segment out different body parts. There are six SMs with the same structure, each of them is responsible for predicting one of the six parts (head, torso, upper arms, lower arms, upper legs and lower legs) as category 1 and the remaining parts as category 2. 
10. Then use the results of the six SMs to teach one small SM. 

The script 
pca_init_caffemodel_weights.py
is for converting the pre-trained weights inside a conventional convolutional layer to the initialization of weights in a pca layer.

Techniques for data augmentation:
The first way of data augmentation is through creating examples which have similar poses to those from the test set. Besides the CNN for segmentation, a model for human pose analysis is trained. Skeleton detection is a more simple task than person part segmentation and there are more training data available. As a result, the predictions from the model for pose analysis tend to be more generalizable. In the proposed method, each image is firstly divided into regions each of which contains one person. For each person in the training set and test set, a vector describing the pose is predicted. Then we try to find the person with the most similar gesture from target regions in test data for each person in the source region from training data. Upon matching each pair of people, we conduct homography transformation on the source regions to make them more similar to the target regions. The ground truth masks for source region are transformed in the same way. The transformations produce fake test images which are similar to test images and with ground truth labels. Some examples are shown in Fig. 1. People are segmented out for clear demonstration.

<img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair1_left.png"/> <img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair1_right.jpg"/> <img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair2_left.jpg"/> <img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair2_right.jpg"/>
<img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair3_left.jpg"/> <img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair3_right.jpg"/>
<img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair4_left.jpg"/> <img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair4_right.jpg"/>
<img width="161" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair5_left.jpg"/> <img width="161" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair5_right.jpg"/>
<img width="289" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair6_left.jpg"/> <img width="161" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig1_pair6_right.jpg"/>
       
Fig. 1. Examples showing 6 pairs of images containing people with similar poses. Within each pair, the left image is from test set, the right one is from training set. We’ve searched the training set to find the person with the most similar pose to each person from test images and conducted homography transformations on the selected regions from training data.
As can be seen from Fig. 1, we’ve generated a fake test set with labels based on training data. We’ve conducted homography transformations to map fake images to real ones based on the coordinates of critical joints. The gap between the fake test set and real test set is surely lower than that between training set and real test set. Experiments will show that the involvement of fake test set during training can improve the performance of segmentation.
To evaluate the similarity in poses, a feature descriptor is developed in our work. The 14 joints are nose, neck, left shoulder, left elbow, left wrist, right shoulder, right elbow, right wrist, left hip, left knee, left ankle, right hip, right knee and right ankle. A center point can be computed by averaging the coordinates of the detected joints and is shown by the red dot in Fig. 2 (a). 
Fig. 2 describes the proposed descriptor, two histograms are used to describe the pose of each person. The sum of Euclidean distances between two pairs of histogram vectors evaluates the similarity in pose between two people. We’ve divided one image into regions each of which contains one person. As a result, multiple images from training data can be used to create one fake test image that is similar to a real test image, as is shown in Pair 6 in Fig. 1. 
Moreover, the existence of other variances, such as rotations, scaling or changes in illumination may reduce the similarity between fake test images and real ones. 

The utilization of test data (without labels) is authorized by transductive learning which is addressed in [2]. Incomplete supervision concerned the situation in which a small amount of labeled data and abundant unlabeled data are available, and the labeled data was insufficient to train a good learner. The unlabeled data are exploited by semi-supervised learning in addition to labeled data to improve learning performance. Transductive learning is a special type of semi-supervised learning and assumes that the available unlabeled data is exactly the test data.
   
<img width="658" height="453" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig2.jpg"/>
Fig. 2. The proposed descriptor for describing poses. (a) The indices of joints. (b) The offsets of different joints from the center. The offsets are normalized with respect to the maximal distance between pairs of joints. The top histogram describes the normalized offsets in the vertical direction while the bottom one describes those in the horizontal direction.


Preprocessed images by step 1 to step 3:

https://drive.google.com/file/d/1GbOFQc7XI3DVLUZpM4dFuXq60fE5OnuO/view?usp=sharing

https://drive.google.com/file/d/1UQRvmh9Kr79SpdzgVQdU1HzPXylmkvpA/view?usp=sharing

In step 3, we need to ensure all identities are with similar scale. If the scales of people in some images are large while those in other images are small, the capacity of the model is not enough. The strategy to ensure the consistency of scale is demonstrated in Fig. 3.

<img width="600" height="300" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/Fig6.png"/>

Fig. 3. For small-scale identities, we firstly use the detection model which is more robust to scale variances to detect the identities and then enlarge the interested area.

We've found that PSPNet-101 [1] can outperform our un-compressed SM and if it is trained on the datasets above and the predictions from the pre-trained PSPNet-101 are used to train our simplified SM, the simplified SM can achieve an accuracy of 76.03% which is higher than as claimed in the paper.
As a result, we've trained PSPNet-101 on our pre-processed images (pre-processed by LM shown above) and get the following models:

segmentation of 1 class in step 1:
https://drive.google.com/file/d/1Ms0ObVzhwH__Jci05Kv-LBja7oDiTVP_/view?usp=sharing

segmentation of 7 classes in step 10:
https://drive.google.com/file/d/1x1hGOTMvm1PZyZbUzTcAb1h2_3UG8-6j/view?usp=sharing

The model can achieve a test accuracy of over 77% and with it as the teacher, a simplified student network can achieve an accuracy of about 76%.
Test list:
https://drive.google.com/file/d/10e_QEpc4izy2XWBNsg-U016in2pIq7sJ/view?usp=sharing

We've also tried to dress the people on the training images with the cloth from people in test images. In that way the test images can be better simulated. The ground truth labels from training images can be mapped with the same mapping function. For instance, the left image comes from the training set while the middle one from test set. The right one is the result with ground truth label.

<img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000176.jpg"/>  <img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2009_003290.jpg"/>  <img width="97" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/3290_dress_176.jpg"/>

Additionally, the exchange of background between training images and test ones are tried:

<img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000277.jpg"/>  <img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000277_0.jpg"/>  <img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000277_1.jpg"/>  <img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000277_2.jpg"/>  <img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000277_3.jpg"/>

<img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000109.jpg"/>  <img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000109_0.jpg"/>  <img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000109_1.jpg"/>  <img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000109_2.jpg"/>  <img width="277" height="208" src="https://github.com/AllenYLJiang/Capacity-Optimization/blob/master/imgs/2008_000109_3.jpg"/> 
 
The experiments on the LIP is based on a more complex model. This is because the small model cannot perform as well on the large dataset. The pre-trained model:
https://drive.google.com/file/d/1yDt0EbwfWLJfljVuKsW7RXMZgc4Wt1wd/view?usp=sharing
https://drive.google.com/file/d/1sPZPKo4ZWMasbrpQMBbWsqkfatl3OLjR/view?usp=sharing
By using domain randomization the performance cn be further improved. 


[1] Zhao, Hengshuang, Jianping Shi, Xiaojuan Qi, Xiaogang Wang, and Jiaya Jia. "Pyramid scene parsing network." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2881-2890. 2017.

[2] Z. H. Zhou, "A brief introduction to weakly supervised learning," National Science Review, vol. 5, no. 1, pp. 44-53, 2017. 
