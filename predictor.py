import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import numpy as np
import pandas as pd
from PC6_encoding import PC_6
import argparse
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU

def main(input_fasta_name, output_csv_name, model_type= None):
    # translate fasta file to PC_6 encode np.array format
    dat = PC_6(input_fasta_name)
    data = np.array(list(dat.values())) 
    # load model ('aug' for AVP+GAN_aug model, 'n' for AVP model)
    model_type = model_type or 'aug'
    # root=os.path.dirname(__file__)
    if model_type == 'aug':
        model = load_model('/home/AI4AVP_predictor/model/PC_6_model_best_weights.h5')
    if model_type == 'n':
        model = load_model('/home/AI4AVP_predictor/model/PC_6_model_n_best_weights.h5')
    # predict
    score = model.predict(data)
    classifier = score>0.5

    # make dataframe
    df = pd.DataFrame(score)
    df.insert(0,'Peptide' ,dat.keys())
    df.insert(2,'Prediction results', classifier)
    df['Prediction results'] = df['Prediction results'].replace({True: 'Yes', False: 'No'})
    df = df.rename({0:'Score'}, axis=1)
    # output csv
    df.to_csv(output_csv_name)

#arg
if __name__ == '__main__':   
    parser = argparse.ArgumentParser(description='PC6 predictor')
    parser.add_argument('-f','--fasta_name',help='input fasta name',required=True)
    parser.add_argument('-o','--output_csv',help='output csv name',required=True)
    parser.add_argument('-m','--model_type',help='model type "aug" of "n"',required=False)
    args = parser.parse_args()
    input_fasta_name = args.fasta_name
    output_csv_name =  args.output_csv
    model_type = args.model_type
    main(input_fasta_name,output_csv_name,model_type)