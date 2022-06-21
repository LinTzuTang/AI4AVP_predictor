# AI4AVP_predictor
![](https://i.imgur.com/HWPjJ4R.png)
### AI4AVP is a sequence-based antiviral peptides (AVP) predictor based on PC6 protein encoding method [[link]](https://github.com/LinTzuTang/PC6-protein-encoding-method) and deep learning.
##### AI4AVP (web-server) is freely accessible at https://symbiosis.iis.sinica.edu.tw/AI4AVP/

### Here we give a quick demo and command usage of our AI4AMP model.  
### 1. quick demo of our PC6 model
##### For quick demo our model, run the command below
```bash 
bash AI4AVP_predictor/test/example.sh
```
##### The input of this demo is 10 peptides (```test/example.fasta```) in [FASTA](https://en.wikipedia.org/wiki/FASTA_format) format.
##### The prediction result (```test/example_output.csv```) below shows prediction scores  and whether the peptide is an AMP in table.
![](https://i.imgur.com/xLjlGHV.png)
### 2. command usage
##### Please make sure your working directory access to  ```AI4AVP_predictor/predictor.py``` and execute command like the example below
```bash
python3 predictor.py -f [input.fasta] -o [output.csv]
```
##### -f : input peptide data in FASTA format
##### -o : output prediction result in CSV 

### 3. Model training source code
##### Our AI4AVP model traing source code is shown in  ```AI4AVP_predictor/model_training```
The model architecture was based on three layers of CNN (filters: (64, 32, 16), kernel_size: (8,8,8)) with rectified linear activation function (ReLU). Every output from the CNN layer was conducted to batch normalization and dropout (rate: (0.5,0.5,0.5)). Finally, there was a fully connected layer (units: 1) with a sigmoid activation function making output values between 0 to 1. 

For model training, we randomly split 10% of training data as a validation dataset and set the batch size to 1000. We focused on validation loss every epoch during model training, then stopped training when the training process was stable, and the validation loss was no longer decreasing. Meanwhile, the model at the epoch with the lowest validation loss was saved as the final best model.


