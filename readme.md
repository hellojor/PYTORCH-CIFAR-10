STEP 1 JMODIFY YOUR ENV (open .bat and change "call activate yourenv")
STEP 2 JUST RUN .bat!!!!!

You can modify argument of train.py.
i.e @python train.py --num_epoch 100 --batch_size 20

train.py	-> start training (-h or --help to look argument)
predict.py 	-> testing
model.py 	-> modify your model
config.py 	-> modeify all the parameter of the network
ProgressBar.py 	-> show the status while training
torchsummary.py -> show the neural network summary
