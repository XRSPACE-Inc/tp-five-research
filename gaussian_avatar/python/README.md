# Gaussian Avatar

This work is using FlashAvatar as code base, and did not finished.


## Installation
environment.yml has everything you needs, but if you hit installation error for submodules, you can just clone https://github.com/graphdeco-inria/gaussian-splatting.git and install it there.


## Example Training data

1. [myslate_11_css_iphone.zip](https://drive.google.com/file/d/1lfxO1sDDyK7_Ic0p5-ssNAxPdEdQ9vDp/view?usp=drive_link) and uncompress into dataset/


## Trains

There're many train file here, just for testing each idea is working. And basically each just using different dataset format.

Note: train.py is base train file, noting chnaged just same as original code.

1. train.py is original code. I've not changed a bit. It uses flame face model to train, and dataset is from metrical-tracker.

2. train_ict.py is same training but replace 3dmm by ictfacekit, dataset is also from metrical-tracker.

3. train_custom.py is using ictfacekit as 3dmm, but dataset is using iphone arkit blendshape.

4. train_custom1.py is using flame face model as 3dmm, but dataset is using iphone arkit blendshape.

5. train_custom_mediapipe.py is using mediapipe blendshape as dataset, flame as 3dmm.

6. train_custom_trans.py is a failed test, I do not remember what it is.

7. train_2d.py is a simple idea trying to reconstruct a 2d video, but not really good result, it uses a 2d video & its pose landmarks as input, a simple sphere as model (not 3dmm). Training data is gone but result stays in dataset/video0/

## Tests

All tests is pair to train file.

Note: the generated avi is broken for some reason... I not fix it anyway.


## Training Data:

1. camera parameter you can gain from [metrical-tracker (mica)](https://github.com/Zielon/metrical-tracker)

2. sigmentation look at: tests/face_parsing.py

3. finally you needs arkit blendshape, I uses livelink face from unreal

    a. after you records you will have a video with blendshape as csv.

    b. dump all image using ffmpeg and put into a folder (name is same as video).

    c. run face_parsing.py (change path in code to where your data is), and put into same folder.

## Export

1. full_export_ict.sh to export model trained with ictfacekit.

2. full_export.sh exports model trained with flame model.

3. note: run in python/ after you done results is in export/outputs/ 

    a. export/outputs/exported0.onnx

    b. export/outputs/exported0_color.bytes

    c. validate files also can find: exported0_input.txt & exported0_output.txt

    d. exported0_input is sample input & exported0_output is expected output.
