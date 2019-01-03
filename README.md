# Capacity-Optimization

The implementation is divided into the following steps:
Step 1 to are for a large network. After being fully trained, the large network is used to teach a small network.
1. Train the large SM to segment out all person parts as one category. This is much simpler than 7 class prediction and the network can roughly segment out all people.
2. Also use LM to predict the skeletons of all people.
3. Based on the results from Step 1 and Step 2, we can roughly segment out the regions containing people and set the backgrounds to zeros. This can reduce the workload of CNNs. 
4. Train and test on the simplified images.
5. Based on the similarity of skeletons, choose some people from training images to simulate those from test images. Especially simulate the cases where people in test images are small, or of hard poses.
6. Data augmentation: Scaling, Rotation, Color(Add a value to the hue channel of the HSV representation) Also we need to random the backgrounds and foregrounds.
