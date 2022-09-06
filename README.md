# Protect your face ！
> Are you always unable to resist popping pimples? This is an algorithm that can use your computer camera to detect whether you are popping pimples. let's maintain a clean and tidy face together.

We provide a detailed implementation method here. By following the steps below, you can run this algorithm directly even on the CPU of your own computer and maintain real-time performance. It should be noted that we provide pre-trained models that can be run directly, but in order to ensure the accuracy of the algorithm, we still recommend that you capture your own videos and train.

## Demo
You can directly use our trained model for testing, but limited by the blogger's special dataset (only contains my own photos, fixed camera background), the accuracy may be terrible. Run the following code to test the pretrained model:
```
python main.py
```

## Train your own model
First, record videos to make your own dataset. We have encapsulated all the steps, you only need to capture two kinds of ".mp4" videos: pop-pimple video (put it in './data/on_face_video') and normal video (put it in './data/no_on_face_video'). It's better to be more than one minute in total duration for each type of video, and the duration of the two videos is similar. Then, run:
```
sh video.sh
```

Your own dataset is finished. Next, train your own model:
```
python train.py
```

Then, replace the "./save_model/best_model/best_model.pth", and you can test it on your computer, run:
 ```
 python main.py
 ```
 
 Good Luck!
