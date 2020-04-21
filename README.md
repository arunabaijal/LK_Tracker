# LK_Tracker
Implementation of the Lucas-Kanade (LK) template tracker

## System Requiremnets
`python3`
`OpenCV (4.2.0)` 

## Run instructions
Download the three datasets to the folder containing track.py. Navigate to that fodler and run<br>
```
python3 track.py --test_video 1 
```

For car dataset, give test_video argument as 1. For bolt dataset, give test_video argument as 2. For DragonBaby dataset, give test_video argument as 3. The default value is 1. 

While program is executing, you can see the output frames in `Output_car / Output_bolt / Output_dragon`. 
After program execution, the resultant video will be saved as `CarResult.mp4 / BoltResult.mp4 / DragonBabyResult.mp4`.
