# coding: utf-8
#
# Filename: classify_logical_relationships.py
# Author: Reto Gubelmann
#
#************************************************************
# Functionality
#
# The script takes in an appropriately formated sequence of two statements
# and classifies it according to their logical relationships as 
# 0 entailment
# 1 contradiction
# 2 neutral
# Takes in a modellist, an argument list (premise \t hypothesis)
# outputs premise \t hypothesis \t class
#
#*************************************************************
# Sample call
#

#
#*************************************************************
# 
# 
#Begin of Program


# Import Modules, etc.

import re
import string
import sys
import torch as torch
from transformers import (BertTokenizer, BertTokenizerFast, BertModel, BertForMaskedLM, AutoModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification, BertForSequenceClassification,
AutoModelForCausalLM, AutoModelForMaskedLM, AutoModelForSeq2SeqLM, T5Tokenizer, T5Config, T5ForConditionalGeneration, pipeline)
from timeit import default_timer as timer
import nltk
import argparse


### ***************************************

# Defining a few functions

### ********************************************


# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument("modelfile", type=str,
                    help="specify the file to extract model information from")

parser.add_argument("taskfile",type=str,
                    help="please specify the dataset-file")

parser.add_argument("masteroutputfile", type=str,
                    help="specify a filename to write the output to")

args = parser.parse_args()


filename_modelfile=args.modelfile
filename_taskfile=args.taskfile
filename_masteroutputfile=args.masteroutputfile

themasteroutputfile=open(filename_masteroutputfile,"w",buffering=1)
print("Modelname|N. Tasks Processed|N. Corr Labels|Precision",  file=themasteroutputfile)


themodelfile=open(filename_modelfile,"r")
next(themodelfile)
thetaskfile=open(filename_taskfile,"r")
next(thetaskfile)

    
# Read model specifica from file

modeltypes=[]
modelnames=[]
tokenizers=[]
printmodelnames=[]
modelpattern=re.compile('([^,]+),([^,]+),([^,]+),([^,]+)')
for line in themodelfile:
    matched=modelpattern.match(line)
    modeltypes.append(matched.group(1))
    modelnames.append(matched.group(2))
    tokenizers.append(matched.group(3))
    printmodelnames.append(matched.group(4))

print(printmodelnames)    
# Read premises and hypotheses in two arrays

premises =[]
hypotheses=[]
true_labels=[]
pairids=[]

premises_hypotheses_pattern=re.compile('(.+)\|(.+)\|(.+)\|(.+)\|(.+)')

#all_tasklines = thetaskfile.readlines()
tasklinecounter=0
for line in thetaskfile:
    tasklinecounter+=1
#    twokflag=0
#    if tasklinecounter % 2000 ==0:
#        twokflag=1
    #print("Now working on taskline:",line)    
    
    matched=premises_hypotheses_pattern.match(line)
    pairids.append(matched.group(1))
    premises.append(matched.group(3))
    hypotheses.append(matched.group(4))
    true_labels.append(matched.group(5))
    
#    if twokflag ==0:
#        continue
    
print("Length of Premises:",len(premises))
print("Length of Hypotheses:",len(hypotheses))
print("Length of True labels:",len(true_labels))
print("\nLast three items of Premises:")
print(premises[-1])
print(premises[-2])
print(premises[-3])
print("\nLast three items of Hypotheses:")
print(hypotheses[-1])
print(hypotheses[-2])
print(hypotheses[-3])
print("\nLast three items of True labels:")
print(true_labels[1])
print(true_labels[2])
print(true_labels[3])

print("\nLast three items of PairIDs:")
print(pairids[-1])
print(pairids[-2])
print(pairids[-3])
# print("Modelmatters:",modeltypes,modelnames,tokenizers,printmodelnames)    


for modelline in range(len(modeltypes)):
    print("\nNow using model with printname",printmodelnames[modelline])
    
    # Specify outputfilename PROBRUNS 
     
    outputfilename="./Outputs/Runs_MaiJune22_Eval_MNLI_GKR/"+printmodelnames[modelline].strip("\n")+".csv"
    print("Writing detailed stats to",outputfilename)
    theoutputfile=open(outputfilename,"w")
    print("PairID|Premise|Hypothesis|Pred_label|True_label|Rightorwrong|Modelname",file=theoutputfile)
    
    # Initialize modelcounters
    mwide_true=dict()
    mwide_sproc=dict()
    
    
    labels_array=[]
    if "Models-" in modelnames[modelline]:
        print("Nice - you're using a custom TF model! Respecting that",file=sys.stderr)
        model = AutoModelForSequenceClassification.from_pretrained(modelnames[modelline],from_tf=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(modelnames[modelline])
    tokenizer = AutoTokenizer.from_pretrained(tokenizers[modelline])
    # Put Model to cuda, if available
    cudaflag = torch.cuda.is_available()
    #cudaflag=0 # check, hack because of memory issues on dgx
    if (cudaflag):
        model.to("cuda:0")
    
    overall_labels=0
    iminusone=0
    
    #for i in range(2000, len(premises),2000):
    for i in range(100, len(premises),100):
        current_premises = premises[iminusone:i]
        current_hypotheses=hypotheses[iminusone:i]
        curr_pairids=pairids[iminusone:i] 
        current_true_labels=true_labels[iminusone:i] 
        #print("Length of current premises:",len(current_premises))
        #print("Length of current hypotheses:",len(current_hypotheses))
        print("Run starting from item",iminusone,"going to",i,"; first premiss-Line is now:",current_premises[0])
        if (cudaflag):
            features = tokenizer(current_premises,current_hypotheses,  padding=True, truncation=True, return_tensors="pt").to("cuda:0")
        else:
            features = tokenizer(current_premises,current_hypotheses,  padding=True, truncation=True, return_tensors="pt")
    
        model.eval()
        with torch.no_grad():
            scores = model(**features).logits
            label_mapping = ['contradiction', 'entailment', 'neutral']
            labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
        
        for item in range(len(labels)):
            
            single_currlab= current_true_labels[item] 
            mwide_sproc[single_currlab] = mwide_sproc.get(single_currlab,0) +1 
             
            trueorfalse=0
            
            if labels[item].strip()==current_true_labels[item].strip():
                trueorfalse=1
                mwide_true[single_currlab] =mwide_true.get(single_currlab,0) +1 
            #else:
                #print("No fit here between label",labels[item].strip(),"and",true_labels[item].strip())
            print(curr_pairids[item],"|",current_premises[item],"|",current_hypotheses[item],"|",labels[item],"|",current_true_labels[item],"|",trueorfalse,"|",modelnames[modelline], file=theoutputfile)
        
        iminusone=i+1
        overall_labels+=len(labels)
    
    theoutputfile.close()
    #print("Truelabels-Dict is:",mwide_true)
    #print("Totalsamples-Dict is:",mwide_sproc)
        
    print("\nOverall precision for model",modelnames[modelline],":",sum(list(mwide_true.values()))/overall_labels,"\n")
    for key in mwide_sproc:
        numbofcorr=0
        if key in mwide_true:
            numbofcorr=mwide_true[key]
        print(printmodelnames[modelline].strip("\n"),"|",mwide_sproc[key],"|",numbofcorr,"|", numbofcorr/mwide_sproc[key],file=themasteroutputfile)
#        print("\nOverall precision:",truecounter/len(labels),"\n", file=themasteroutputfile)
#    premises =[]
#    hypotheses=[]
#    true_labels=[]
#    syllogism_types=[]        
themasteroutputfile.close()
    