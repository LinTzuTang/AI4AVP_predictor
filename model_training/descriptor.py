import numpy as np
import pandas as pd
import re
import math
from Bio import SeqIO
from collections import Counter
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

# read fasta as dict
def read_fasta(fasta_fname,length=None):
    r = dict()
    for record in SeqIO.parse(fasta_fname, 'fasta'):
        idtag = str(record.id)
        seq = str(record.seq)[:length]
        r[idtag] = seq
    return r

def globaldescriptor(fasta_fname):
    r = read_fasta(fasta_fname)
    desc = GlobalDescriptor(list(r.values()))
    desc.calculate_all(amide=True)
    # desc.featurenames (['Length','MW','Charge','ChargeDensity','pI','InstabilityInd','Aromaticity','AliphaticInd','BomanInd','HydrophRatio'])
    return desc.descriptor

def cal_physc_df(fasta_fname, length=None):
    # read fasta to df 
    r = read_fasta(fasta_fname,length)
    seq_df = pd.DataFrame(data= r.items(), columns=["Id", "Sequence"])
    
    # Calculating Global descriptor ('Length','MW','Charge','ChargeDensity','pI','InstabilityInd','Aromaticity','AliphaticInd','BomanInd','HydrophRatio')
    desc = GlobalDescriptor(seq_df['Sequence'].values)
    desc.calculate_all(amide=True)
    seq_df[desc.featurenames] = desc.descriptor

    # Calculating Peptide descriptor ('hydrophobic moment', 'hydrophobicity', 'Transmembrane Propensity', 'Alpha Helical Propensity')
    # hydrophobic moment (Assume all peptides are alpha-helix)
    desc = PeptideDescriptor(seq_df['Sequence'].values, 'eisenberg')
    desc.calculate_moment(window=1000, angle=100, modality='max')
    seq_df['Hydrophobic Moment'] = desc.descriptor

    # "Hopp-Woods" hydrophobicity
    desc = PeptideDescriptor(seq_df['Sequence'].values, 'hopp-woods')
    desc.calculate_global()
    seq_df['Hydrophobicity'] = desc.descriptor

    # Energy of Transmembrane Propensity
    desc = PeptideDescriptor(seq_df['Sequence'].values, 'tm_tend')
    desc.calculate_global()
    seq_df['Transmembrane Propensity'] = desc.descriptor

    # Levitt_alpha_helical Propensity
    desc = PeptideDescriptor(seq_df['Sequence'].values, 'levitt_alpha')
    desc.calculate_global()
    seq_df['Alpha Helical Propensity'] = desc.descriptor
    return seq_df
    

def min_sequence_length(fastas_list):
    minLen = 1000000
    for i in fastas_list:
        if minLen > len(re.sub('-', '', i[1])):
            minLen = len(re.sub('-', '', i[1]))
    return minLen

def paac(fastas_list, lambdaValue=30, w=0.05, **kw):
    if min_sequence_length(fastas_list) < lambdaValue + 1:
        print('Error: all the sequence length should be larger than the lambdaValue+1: ' + str(lambdaValue + 1) + '\n\n')
        return 0

    dataFile = '../data/PAAC.txt' 
    with open(dataFile) as f:
        records = f.readlines()
    AA = ''.join(records[0].rstrip().split()[1:])
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    AAProperty = []
    AAPropertyNames = []
    for i in range(1, len(records)):
        array = records[i].rstrip().split() if records[i].rstrip() != '' else None
        AAProperty.append([float(j) for j in array[1:]])
        AAPropertyNames.append(array[0])

    AAProperty1 = []
    for i in AAProperty:
        meanI = sum(i) / 20
        fenmu = math.sqrt(sum([(j-meanI)**2 for j in i])/20)
        AAProperty1.append([(j-meanI)/fenmu for j in i])

    encodings = []
    header = ['#']
    for aa in AA:
        header.append('Xc1.' + aa)
    for n in range(1, lambdaValue + 1):
        header.append('Xc2.lambda' + str(n))
    encodings.append(header)
    
    def Rvalue(aa1, aa2, AADict, Matrix):
        return sum([(Matrix[i][AADict[aa1]] - Matrix[i][AADict[aa2]]) ** 2 for i in range(len(Matrix))]) / len(Matrix)

    for i in fastas_list:
        name, sequence = i[0], re.sub('-', '', i[1])
        code = [name]
        theta = []
        for n in range(1, lambdaValue + 1):
            theta.append(
                sum([Rvalue(sequence[j], sequence[j + n], AADict, AAProperty1) for j in range(len(sequence) - n)]) / (
                len(sequence) - n))
        myDict = {}
        for aa in AA:
            myDict[aa] = sequence.count(aa)
        code = code + [myDict[aa] / (1 + w * sum(theta)) for aa in AA]
        code = code + [(w * j) / (1 + w * sum(theta)) for j in theta]
        encodings.append(code)
        encodings_df = pd.DataFrame(data= encodings[1:], columns=encodings[0])
    return encodings_df

def aac(fastas_list, **kw):
    #AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    #AA = 'ARNDCQEGHILKMFPSTWYV'
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    header = ['#']
    for i in AA:
        header.append(i)
    encodings.append(header)

    for i in fastas_list:
        name, sequence = i[0], re.sub('-', '', i[1])
        count = Counter(sequence)
        for key in count:
            count[key] = count[key]/len(sequence)
        code = [name]
        for aa in AA:
            code.append(count[aa])
        encodings.append(code)
        encodings_df = pd.DataFrame(data= encodings[1:], columns=encodings[0])
    return encodings_df

def dpc(fastas_list, **kw):
    #AA = kw['order'] if kw['order'] != None else 'ACDEFGHIKLMNPQRSTVWY'
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#'] + diPeptides
    encodings.append(header)

    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i

    for i in fastas_list:
        name, sequence = i[0], re.sub('-', '', i[1])
        tmpCode = []
        tmpCode = [0] * 400
        for j in range(len(sequence) - 2 + 1):
            tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] = tmpCode[AADict[sequence[j]] * 20 + AADict[sequence[j+1]]] +1
        if sum(tmpCode) != 0:
            tmpCode = [i/sum(tmpCode) for i in tmpCode]
        encodings.append(tmpCode)
    return encodings[1:]

def aaindex(fastas_list, **kw):

    AA = 'ARNDCQEGHILKMFPSTWYV'

    fileAAindex = '../data/AAindex.txt'
    with open(fileAAindex) as f:
        records = f.readlines()[1:]

    AAindex = []
    AAindexName = []
    for i in records:
        AAindex.append(i.rstrip().split()[1:] if i.rstrip() != '' else None)
        AAindexName.append(i.rstrip().split()[0] if i.rstrip() != '' else None)

    index = {}
    for i in range(len(AA)):
        index[AA[i]] = i

    encodings = []
    header = ['#']
    for pos in range(1, len(fastas_list[0][1]) + 1):
        for idName in AAindexName:
            header.append('SeqPos.' + str(pos) + '.' + idName)
    encodings.append(header)

    for i in fastas_list:
        name, sequence = i[0], i[1]
        code = []
        for aa in sequence:
            if aa == '-':
                for j in AAindex:
                    code.append(0)
                continue
            for j in AAindex:
                code.append(j[index[aa]])
        encodings.append(code)
    return encodings[1:]

def ennavia_encoding(fasta_fname):
    physc_list = np.array(cal_physc_df(fasta_fname).iloc[:,4:])
    r = read_fasta(fasta_fname)
    paac_list = np.array(paac(list(r.items()),lambdaValue=9).iloc[:,1:])
    aac_list = np.array(aac(list(r.items())).iloc[:,1:])
    dpc_list = np.array(dpc(list(r.items())))
    encoding_array = np.concatenate((physc_list,paac_list,aac_list,dpc_list),axis=1)
    return encoding_array
