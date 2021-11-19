# U-Net: Convolutional Network for Ship Segmentation

<h1> Project scheme</h1>
<p>.
<p>├── model
<p>│   ├── unet.py                 # defines U-Net class
<p>│   └── utils.py                # dice loss for model
<p>├── tools                     
<p>│   └── tool.py                 # image train/test generator and preprocesing csv 
<p>├── main.py                     # main script to run
<p>├── predict.py                  # Predict result (show 2x2 img and predictive mask)
<p>├── requirements.txt            # needed libraries
<p>└── seg_model_weights_best.hdf5 # best model weights

<h1> Instructions</h1>
Define your config in main.py file:

<p>train_path =              # define path to train images ..../train_v2
<p>segment_path =            # define path to segmentation csv .../train_ship_segmentations_v2.csv

<h2> Model Training</h2>
To start training model, run main.py script.

It will create one more csv file for balance images for training, then it will only open and read this file, don't need processing again.
Than will be created training generator that create BATCH_SIZE examples of train (image, mask).
Size of image that it create will be (256, 256, 3) and mask (256, 256)
After it will get model from model.unet, and build it, get all needed callbacks.
An of course fit it, using dice loss and same metrics from model.utils
During fitting it will save best weight for model, and if SAVE_FINALE_MODEL = True, save final model.
  
 <h2> Model Predict</h2>
   Define your config in predict.py:
   
   test_path =               # define path to test images ..../test_v2
   
   To predict model, you need to run predict.py
   It will show you grafics, that has test images size 2x2 and there predictive mask
   When you close graphic, it will generate 4 more pictures and so on
   Until you stop predict.py using (ctr+F2) or stop buttom
   predict.py will create test generator, that generate 4 image, predict there mask and show, using matplotlip

   



