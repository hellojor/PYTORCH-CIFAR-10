# Train and Test PYTORCH-CIFAR-10(ACC:92.26%)

## Usage
```
train.py        start training (-h or --help to look argument)
predict.py      testing
model.py        modify your model
config.py       modeify all the parameter of the network
ProgressBar.py  show the status while training
```
<br>

## My Enviroment
- Anaconda
- Python 3.8.10
- PyTorch 1.10.0

<br>

## Training
  You can start training with `python train.py`
  Also can modify the parameter like `python train.py --lr 0.01 --num_epoch 100 --batch_size 20`
  Just use `python train.py -h` to check it!!!

<br>

## Testing
  Just run `python predict.py`

<br>

## Simple train and test
  STEP 1 MODIFY YOUR ENV (open .bat and change "call activate yourenv")<br>
  STEP 2 JUST RUN `run_train_and_test.bat` (or `run_test.bat`)!!!!!<br>
  You can modify argument of `train.py`.<br>
  `i.e @python train.py --num_epoch 100 --batch_size 20`

<br>

## Accuracy
  Accuracy: 92.26 %<br>
  Accuracy of plane : 92 %<br>
  Accuracy of car   : 97 %<br>
  Accuracy of bird  : 89 %<br>
  Accuracy of cat   : 83 %<br>
  Accuracy of deer  : 92 %<br>
  Accuracy of dog   : 86 %<br>
  Accuracy of frog  : 94 %<br>
  Accuracy of horse : 95 %<br>
  Accuracy of ship  : 95 %<br>
  Accuracy of truck : 94 %<br>

<br>

## Result
  [result](https://github.com/hellojor/PYTORCH-CIFAR-10/blob/main/result.jpg)

## Conclusion
  Remind me if there is any bug on it. 
  THX FOR READING.
