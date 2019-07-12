# Seq2Seq LSTM Model
This model is trained to set benchmark on traffic flow dataset of kaggle
## Running model
* Training model
```bash
python train.py \
--batch_size \
--time_step
Note: need to change which sensor to run inside the code of train.py
```

* testing model
```bash
python test.py --model_path
```

