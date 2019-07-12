# Seq2Seq LSTM Model
This model is trained to set benchmark on traffic flow dataset of kaggle

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
1. Dataset:
It contains 2 files: 
- oct_data.txt {contains 3 column: 1st is time(ddhhmm), 2nd is sensor no and 3rd is traffic count}
- lstminput.txt {contains 35 column(35 sensors): each column sensor number is in increasing order i.e. column 1: sensor no 0,....... column i: sensor no i-1} 

2. script
- Preprocess folder
- train.py 
- test.py
3. trained_models (contains trained models) 


## Running model 
* Training model
```bash
cd script/
python train.py \
--batch_size \
--time_step
Note:
- I have used batch size 10 and timestep 3 
- need to change which sensor to run inside the code of train.py
```

* testing model
```bash
python test.py --model_path
```

