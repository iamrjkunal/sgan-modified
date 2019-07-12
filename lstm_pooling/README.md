# Seq2Seq LSTM Model with pooling module
This model is trained to on traffic flow dataset of kaggle 

## Setup
All code was developed and tested on Ubuntu 16.04 with Python 3.5 and PyTorch 0.4.

You can setup a virtual environment to run the code like this:

```bash
python3 -m venv env               # Create a virtual environment
source env/bin/activate           # Activate virtual environment
pip install -r requirements.txt   # Install dependencies
echo $PWD > env/lib/python3.5/site-packages/sgan.pth  # Add current directory to python path
# Work for a while ...
deactivate  # Exit virtual environment
```


## Folder Structure
1. dataset:
It contains 2 files: 
- oct_data.txt {contains 3 column: 1st is time(ddhhmm), 2nd is sensor no and 3rd is traffic count}
- lstminput.txt {contains 35 column(35 sensors): each column sensor number is in increasing order i.e. column 1: sensor no 0,....... column i: sensor no i-1} 

2. script
- Preprocess folder
- trained_model folder
#### Development versions
- dev.py (sample k= 5 sensors time step = 10) : architecture 1
- dev1.py : (sample k= 5 sensors time step = 3) architecture 2
- dev2.pym : (sample k= 5 sensors time step = 5) architecure 1
- dev3.py : (sample k= 34 sensors time step = 5) architecture 1
- dev_def.py : (sample k= 34 sensors time step = 3 and changed dimensions of hyperparameters )architecture 1
#### Final Versions
- dev_final_train.py (sample k= 5 sensors time step = 10) : architecture 1
- dev_final_test.py (sample k= 5 sensors time step = 10) :architecture 1
```
Note:
- architecture 1 works
- architecture 2 fails
- Difference between architecture 1 & 2 is only in decoder part.
```
## Running model 
* Training model
```bash
cd script/
python dev_final_train.py \
--num_sensors \
--sensor_no
Note:
- num_sensors = no. of sensors you want to sample
- sensor_no = for which sensor we want to predict 
```

* testing model
```bash
python dev_final_test.py \
--num_sensors \
--sensor_no
```

