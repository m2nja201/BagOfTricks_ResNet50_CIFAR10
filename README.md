# 💼 BagOfTricks_ResNet50_CIFAR10
experimental code of **``Bag of Tricks for Image Classification with CNN``**

## ⚙️ Setting
- **Dataset** : CIFAR-10
- **Backbone Model** : ResNet50

## 📕 How to
1. Select the train file you want to run.
2. Uncomment the techniques you want to experiment with.
3. Execute the following command:
   
```
python {train file name} --name="{result file name}" > {result file name}.txt
```

## 🎣 Experiment
- Scheduler (CosineAnnealing)
- Warm up
- Drop out
- Weight Decay
- Label Smoothing
- Knowledge Distallation
- Mix up

## 🔍 Result and Conclusion
> I used a part of the PowerPoint presentation I had made as the results section.
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/6ec53d63-a61c-4ba8-afc5-032e3e96e893)
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/3e702269-87a2-416b-a922-5244c8d2560a)
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/4bf7ebd6-1f63-41fe-b12c-013a87a3145b)
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/88e165b3-6759-4c05-ab16-82c8be6643fa)
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/6671af1f-ba37-48b9-a2db-0a8befc46b16)
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/2387aa1a-576c-48df-9d17-264546ffc9cc)
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/861fcb6a-d085-46a8-9cc9-ad508813462e)
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/c3c4cfcd-832f-437e-9bac-5407b1b148de)
![image](https://github.com/m2nja201/BagOfTricks_ResNet50_CIFAR10/assets/80443295/bbbec0f3-d9e2-4833-9175-0d01ab125168)

## 🐨 To Do
- Proceed with ``zero gamma`` and ``knowledge distillation`` for the results with the best performance.

## 🖇️ Reference
1. Tong He, Zhi Zhang, Hang Zhang, Zhongyue Zhang, Junyuan Xie, and Mu Li. Bag of tricks for image classifica-tion with convolutional neural networks. IEEE/CVF conference on computer vision and pattern recognition, pages 558–567, 2019.
2. Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. Deep Residual Learning for Image Recognition. arXiv:1512.03385.


