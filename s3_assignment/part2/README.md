code reefernce  https://github.com/Srinivaskolli45/srinivas/tree/main/s3_assignment/s3_assignment.iypb

Model :

Requirement already satisfied: torchsummary in /usr/local/lib/python3.9/dist-packages (1.5.1)
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 28, 28]             160
              ReLU-2           [-1, 16, 28, 28]               0
       BatchNorm2d-3           [-1, 16, 28, 28]              32
            Conv2d-4           [-1, 16, 28, 28]           2,320
              ReLU-5           [-1, 16, 28, 28]               0
       BatchNorm2d-6           [-1, 16, 28, 28]              32
         MaxPool2d-7           [-1, 16, 14, 14]               0
           Dropout-8           [-1, 16, 14, 14]               0
            Conv2d-9           [-1, 16, 14, 14]           2,320
             ReLU-10           [-1, 16, 14, 14]               0
      BatchNorm2d-11           [-1, 16, 14, 14]              32
           Conv2d-12           [-1, 16, 14, 14]           2,320
             ReLU-13           [-1, 16, 14, 14]               0
      BatchNorm2d-14           [-1, 16, 14, 14]              32
        MaxPool2d-15             [-1, 16, 7, 7]               0
          Dropout-16             [-1, 16, 7, 7]               0
           Conv2d-17             [-1, 16, 7, 7]           2,320
             ReLU-18             [-1, 16, 7, 7]               0
      BatchNorm2d-19             [-1, 16, 7, 7]              32
           Conv2d-20             [-1, 32, 7, 7]           4,640
             ReLU-21             [-1, 32, 7, 7]               0
      BatchNorm2d-22             [-1, 32, 7, 7]              64
        MaxPool2d-23             [-1, 32, 3, 3]               0
          Dropout-24             [-1, 32, 3, 3]               0
           Conv2d-25             [-1, 10, 3, 3]           2,890
             ReLU-26             [-1, 10, 3, 3]               0
      BatchNorm2d-27             [-1, 10, 3, 3]              20
        AvgPool2d-28             [-1, 10, 1, 1]               0
          Dropout-29             [-1, 10, 1, 1]               0
           Linear-30                   [-1, 10]             110
================================================================
Total params: 17,324
Trainable params: 17,324
Non-trainable params: 0

logs:

Test set: Average loss: 0.0601, Accuracy: 9820/10000 (98%)

loss=0.2195345014333725 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.33it/s]

Test set: Average loss: 0.0412, Accuracy: 9867/10000 (99%)

loss=0.1670299768447876 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.84it/s]

Test set: Average loss: 0.0339, Accuracy: 9887/10000 (99%)

loss=0.08429580181837082 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.66it/s]

Test set: Average loss: 0.0320, Accuracy: 9900/10000 (99%)

loss=0.09568411111831665 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.27it/s]

Test set: Average loss: 0.0260, Accuracy: 9930/10000 (99%)

loss=0.056685131043195724 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.68it/s]

Test set: Average loss: 0.0316, Accuracy: 9910/10000 (99%)

loss=0.1363476663827896 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.82it/s]

Test set: Average loss: 0.0236, Accuracy: 9929/10000 (99%)

loss=0.06697998195886612 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 23.30it/s]

Test set: Average loss: 0.0246, Accuracy: 9922/10000 (99%)

loss=0.12657243013381958 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.78it/s]

Test set: Average loss: 0.0237, Accuracy: 9927/10000 (99%)

loss=0.04876984283328056 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 24.73it/s]

Test set: Average loss: 0.0218, Accuracy: 9939/10000 (99%)

loss=0.03758468106389046 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.50it/s]



Test set: Average loss: 0.0268, Accuracy: 9922/10000 (99%)

loss=0.028737055137753487 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.65it/s]

Test set: Average loss: 0.0247, Accuracy: 9930/10000 (99%)

loss=0.13235586881637573 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.62it/s]

Test set: Average loss: 0.0252, Accuracy: 9924/10000 (99%)

loss=0.06626477092504501 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.57it/s]

Test set: Average loss: 0.0216, Accuracy: 9938/10000 (99%)

loss=0.026007361710071564 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 24.66it/s]

Test set: Average loss: 0.0223, Accuracy: 9935/10000 (99%)

loss=0.034668222069740295 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.91it/s]

Test set: Average loss: 0.0270, Accuracy: 9927/10000 (99%)

loss=0.01554136723279953 batch_id=468: 100%|██████████| 469/469 [00:19<00:00, 23.47it/s]

Test set: Average loss: 0.0215, Accuracy: 9937/10000 (99%)

loss=0.030078327283263206 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.06it/s]

Test set: Average loss: 0.0215, Accuracy: 9939/10000 (99%)

Results:
Build the simple network with batch normalization, relu and dropout  and achieved 99 % accuracy

