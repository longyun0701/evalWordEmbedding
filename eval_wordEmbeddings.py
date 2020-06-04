# -*- coding: utf-8 -*-
"""
The module is to evaluation word embeddings
Benchmarks can be (1) word similarity (2) word analogy (3) outlier detection
and can be specified by user in the test_data file
If not specified, the benchmark is just how many words have representation, or 
(4) existence checking

Created on Wed Jun  3 22:07:51 2020
Version 1, completed on Thu Jun  4 17:16:01 2020
 
@author: Dr. Long Yun, longyun0701@hotmail.com

"""

import numpy as np
import sys
import os

###############################################################
# Part 1: read in model parameters for 4 word embeddings:     #
# namely, 'Word2Vec_SG', 'Word2Vec_CBOW', 'FastText', 'GloVe' #
###############################################################

model_names = []      # A list of names for all models
model_dicts = []      # A list for all models: each a {word: vector} dict 
model_keys = []       # A list for all models: each an ndarray of all vocabulary (shape = n_words)
model_vectors = []    # A list for all models: each an ndarray of all word vectors (shape = n_words, n_dims)

dashline = ''.center(60,'-')
print(dashline + '\nLoading model parameters...\n' + dashline)   

with open('models/model_lib.txt', 'r') as f: # model_lib.txt lists the name and parameter filename for all models
    model_lib = f.readlines()
    
    
for model in model_lib:
    model_name, model_file = tuple(model.strip().split(':'))
    model_names.append(model_name)
    wv = np.load('models/'+model_file)
    w = wv['w'].copy()
    v = wv['v'].copy()
    del wv
    model_keys.append(w)
    model_vectors.append(v)
    wv = {w[ikey]:v[ikey] for ikey in range(len(w))}
    model_dicts.append(wv)
    print(model_name.rjust(15) + ": word embedding parameters loaded OK!")
print(dashline+'\nAll model parameters have been loaded!\n'+dashline+'\n')


##################################################################
# Part 2: Function definitions for 3 types of Evaluation :       #
# namely, 'word similarity', 'word analogy', 'outlier detection' #
# if benchmark not specified, then 'existence test' will be done #
##################################################################

# Part 2.1 Similarity test function

def similarityTest(test_data):
    """
    Word Similarity test: to compare the 4 word embeddings (parameters from global variables) 
        for their performance on the word similarity prediction
        The test data should be a heading line, followed by lines of questions,
            each of which contain 2 words and a human-labeled similarity score
    
    Parameter(s):
        test_data: a txt file, whose 1st line indicate 'similarity' test, and the scale of the similarity score
                   From 2nd line, each line contains 4 words and a human-labeled similarity score
                   Any line NOT starting with an English letter will be ignored
                   
    Return(s):
        best_model: index of the model which is evaluated the 'best' among all embeddings
        Xm: For the 'best' model, X% is the fraction of the existing representations in all test questions 
        Ym: For the 'best' model, Y% is the fraction of a 'good' representation when a representation exist
            ( a 'good' representation is interpreted as: |pred_similarity - human_similarity| < 0.2 )
        The model with highest X*Y is the 'best' model 
    
    Main Reference(s):
        Bin Wang, Angela Wang, Fenxiao Chen, Yuncheng Wang, C.-C. Jay Kuo, 2019
            Evaluating Word Embedding Models: Methods and Experimental Results
        
    """
    print('model'.rjust(15), 'n_test'.rjust(8), 'n_avail (X%)'.center(18), 'n_good (Y%)'.center(18))
    print(dashline)
    
    H_scale = float(test_data[0].strip().split()[1]) # Get the scale of human-labeled similarity score
    best_model = 0
    XYm = 0
    for i_wv in range(len(model_names)):             # Loop over all models
        wv_name = model_names[i_wv]
        wv_dict = model_dicts[i_wv]
        n_test, n_avail, n_good = 0, 0, 0           # Accumulating for #_test, #_existing repr, #_good repr
        
        for test in test_data[1:]:                  # For each model, loop over all test questions
            if not test[0].isalpha(): continue
            n_test += 1
            word1, word2, H_sim = tuple(test.strip().split())
            H_sim = float(H_sim)/H_scale

            if word1 in wv_dict and word2 in wv_dict:  
                # Only when both words in a question exist in a model, we consider repr exists
                n_avail += 1
                wv1 = wv_dict[word1]
                wv2 = wv_dict[word2]
                
                # Check cosine similarity of two word vectors, if close to human label, count as a good one
                pred_sim = wv1.dot(wv2) / np.sqrt(np.sum(wv1**2) * np.sum(wv2**2)) 
                if np.abs(H_sim - pred_sim) < 0.2:
                    n_good += 1
        
        X, Y = n_avail/n_test*100, n_good/n_avail*100
        if X*Y > XYm:  # update the 'best' model if the current one is 'better'
            XYm = X*Y
            Xm,Ym = X, Y
            best_model = i_wv
        print(wv_name.rjust(15), str(n_test).rjust(8), str(n_avail).rjust(8),'(%5.1f' % X +'%)', str(n_good).rjust(8), '(%5.1f' % Y +'%)')
    return best_model, Xm, Ym



# Part 2.2 Analogy test function

def analogyTest(test_data):
    """
    Word Analogy test: to compare the 4 word embeddings (parameters from global variables) 
        for their performance on the word analogy prediction
        The test data should be a heading line, followed by lines of questions,
            each of which contain 4 words, namely a, a*, b, b*,
            where the logical relation of a to a* is similar as that of b to b*
            The question is: given a, a* and b, can the models predict correct b*?
    
    Parameter(s):
        test_data: a txt file, whose 1st line indicate 'analogy' test
                   From 2nd line, each line contains 4 words, a, a*, b, b* with analogy relations
                   Any line NOT starting with an English letter will be ignored
    Return(s):
        best_model: index of the model which is evaluated the 'best' among all embeddings
        Xm: For the 'best' model, X% is the fraction of the existing representations in all test questions 
        Ym: For the 'best' model, Y% is the fraction of a 'good' representation when a representation exist
            ( a 'good' representation is interpreted as: pred_b* == human_b* )
        The model with highest X*Y is the 'best' model 
    
    Main Reference(s):
        Bin Wang, Angela Wang, Fenxiao Chen, Yuncheng Wang, C.-C. Jay Kuo, 2019
            Evaluating Word Embedding Models: Methods and Experimental Results
        
    """
    
    print('Warning: Analogy Test is computationally expensive.')
    print('It may take 3-10 min per model, or a total of 15-30 min for all 4 models.\n')
    print('model'.rjust(15), 'n_test'.rjust(8), 'n_avail (X%)'.center(18), 'n_good (Y%)'.center(18))
    print(dashline)

    best_model = 0
    XYm = 0

    for i_wv in range(len(model_names)):             # Loop over all models
        wv_name = model_names[i_wv]
        wv_dict = model_dicts[i_wv]
        wv_vectors = model_vectors[i_wv]
        wv_keys = model_keys[i_wv]

        n_test, n_avail, n_good = 0, 0, 0           # Accumulating for #_test, #_existing repr, #_good repr
        
        for test in test_data[1:]:                  # For each model, loop over all test questions
            if not test[0].isalpha(): continue 
            n_test += 1
            word1, word2, word3, word4 = tuple(test.strip().split())

            if word1 in wv_dict and word2 in wv_dict and word3 in wv_dict and word4 in wv_dict:
                
                # Only when all words in a question exist in a model, we consider repr exists
                # We apply 3CosAdd, using Eq(9) of Ref: Wang, "Evaluating word embedding models", 
                # as 3CosMul or Eq(10) gave poorer performance
                
                n_avail += 1
                wv1, wv2, wv3 = wv_dict[word1], wv_dict[word2], wv_dict[word3]
                V = wv2 - wv1 + wv3
                argmax = np.argmax(np.sum(wv_vectors*V,axis = 1) / np.sqrt(np.sum(wv_vectors**2,axis = 1) * np.sum(V**2)) )
                if wv_keys[argmax] == word4:
                    n_good += 1
        X, Y = n_avail/n_test*100, n_good/n_avail*100
        if X*Y > XYm:  # update the 'best' model if the current one is 'better'
            XYm = X*Y
            Xm,Ym = X, Y
            best_model = i_wv
        print(wv_name.rjust(15), str(n_test).rjust(8), str(n_avail).rjust(8),'(%5.1f' % X +'%)', str(n_good).rjust(8), '(%5.1f' % Y +'%)')
    return best_model, Xm, Ym


# Part 2.3 Outlier test function

def outlierTest(test_data):
    """
    Outlier detection test: to compare the 4 word embeddings (parameters from global variables) 
        for their performance on the outlier detection prediction
        The test data should be a heading line, followed by lines of questions,
            each of which contain a group of (a few) words and a number, 
            the number is 1-base index of 'outlier' word, which is least compatible in the group
    
    Parameter(s):
        test_data: a txt file, whose 1st line indicate 'outlier' test
                   From 2nd line, each line contains a few words and a number (the answer)
                   Any line NOT starting with an English letter will be ignored
    Return(s):
        best_model: index of the model which is evaluated the 'best' among all embeddings
        Xm: For the 'best' model, X% is the fraction of the existing representations in all test questions 
        Ym: For the 'best' model, Y% is the fraction of a 'good' representation when a representation exist
            ( a 'good' representation is interpreted as the pred_outlier == human_outlier )
        The model with highest X*Y is the 'best' model 
    
    Main Reference(s):
        Bin Wang, Angela Wang, Fenxiao Chen, Yuncheng Wang, C.-C. Jay Kuo, 2019
            Evaluating Word Embedding Models: Methods and Experimental Results
        But the above paper refers to:
            Camacho-Collados, Navigli, 2016, Find the word that does not belong:
            A Framework for an Intrinsic Evaluation of Word Vector Representations
        
    """    
    print('model'.rjust(15), 'n_test'.rjust(8), 'n_avail (X%)'.center(18), 'n_good (Y%)'.center(18))
    print(dashline)

    best_model = 0
    XYm = 0
    
    for i_wv in range(len(model_names)):             # Loop over all models
        wv_name = model_names[i_wv]
        wv_dict = model_dicts[i_wv]
        n_test, n_avail, n_good = 0, 0, 0           # Accumulating for #_test, #_existing repr, #_good repr

        for test in test_data[1:]:                  # For each model, loop over all test questions
            if not test[0].isalpha(): continue 
            n_test += 1

            words = test.strip().split()            # Get the word group
            outlier_ans = int(words.pop())-1        # Get the human-labeled outlier index, change to 0-based

            # Check if all words in the question exist in the model
            unavail = False
            for word in words:
                if word not in wv_dict:
                    unavail = True
                    break
            if unavail: continue   # If not all in, considered no representation exists
            n_avail += 1
            n_words = len(words)
            
            # Calculate all the pair-wise similarity, and saved in a dictionary
            sim_dict = {}
            for i in range(n_words-1):
                wv1 = wv_dict[words[i]]
                for j in range(i+1,n_words):
                    wv2 = wv_dict[words[j]]
                    sim_ij = wv1.dot(wv2) / np.sqrt(np.sum(wv1**2) * np.sum(wv2**2))
                    sim_dict[(i,j)] = sim_ij

            max_compact = 0
            outlier_pred = 0
            for i_word in range(n_words):  # Test if a word is removed, whether the compactness is highest
                compact = 0
                for pair in sim_dict:
                    if i_word not in pair:
                        compact += sim_dict[pair]
                if compact > max_compact:
                    max_compact = compact
                    outlier_pred = i_word
            if outlier_pred == outlier_ans:
                n_good += 1
        X, Y = n_avail/n_test*100, n_good/n_avail*100
        if X*Y > XYm:  # update the 'best' model if the current one is 'better'
            XYm = X*Y
            Xm,Ym = X, Y
            best_model = i_wv
        print(wv_name.rjust(15), str(n_test).rjust(8), str(n_avail).rjust(8),'(%5.1f' % X +'%)', str(n_good).rjust(8), '(%5.1f' % Y +'%)')
    return best_model, Xm, Ym



# Part 2.4 Existence test function

def existTest(test_data):
    """
    Word Existence test: to compare the 4 word embeddings (parameters from global variables) 
        for how many words in test data have representations
        The test data, if not specify a benchmark in the heading line,
            is considered as a natural language article.
            Only words of pure alphatetic (without any special char) are considered.
    
    Parameter(s):
        test_data: a txt file, whose 1st line DOES NOT indicate a benchmark.
                   The content are split into segments by white space. 
                   Any segments with special characters will be ignored.
    Return(s):
        best_model: index of the model which is evaluated the 'best' among all embeddings
        Xm: For the 'best' model, X% is the fraction of the existing representations in all test questions
        The model with highest X is the 'best' model 
    
      
    """
    
    print('model'.rjust(15), 'n_test'.rjust(8), 'n_avail (X%)'.center(18))
    print(dashline)

    best_model = 0
    Xm = 0

    for i_wv in range(len(model_names)):             # Loop over all models
        wv_name = model_names[i_wv]
        wv_dict = model_dicts[i_wv]

        n_test, n_avail = 0, 0              # Accumulating for #_test, #_existing repr
        
        for line in test_data:                  # For each model, loop over all text lines
            line = line.strip().split()
            for word in line:
                if word.isalpha():
                    n_test += 1
                    if word.lower() in wv_dict:
                        n_avail += 1
                        
        X = n_avail/n_test*100
        if X > Xm:  # update the 'best' model if the current one is 'better'
            Xm = X
            best_model = i_wv
        print(wv_name.rjust(15),str(n_test).rjust(8), str(n_avail).rjust(8),'(%5.1f' % X +'%)')
    return best_model, Xm, Xm  # For consistence with other testing


##################################################################
# Part 3: Read in test data file, evaluate models, and report    #
##################################################################

# Mapping the recognizable string for test type with the testing functions
tests = {'!similarity': similarityTest,
         '!analogy': analogyTest,
         '!outlier': outlierTest,
         '!existence': existTest}

if len(sys.argv) == 2:           # Test data file must be input as argv[1]
    path = os.getcwd() + '/testdata'
    all_files = [f for f in os.listdir(path)]
    if sys.argv[1] in all_files:
        datafile = sys.argv[1]
        with open('testdata/'+ datafile, 'r') as f:
            test_data = f.readlines()
        dashline = ''.center(60,'-')
        test_type = test_data[0].split()[0]
        
        if test_type not in tests:
            test_type = '!existence'
        
        print(dashline +'\nPerforming ' + test_type.upper()[1:] + ' test (on "testdata/'+sys.argv[1] +'") ...\n' +dashline)  

        best_model, X, Y = tests[test_type](test_data)
        # For existence checking, an exisiting repr is a good repr, i.e., X=Y
            
        print(dashline)
        print('The Best model on the given data set is: '+ model_names[best_model])
        print('As there are representations for X = %.1f' % X  +'%' +' of the testing questions,')
        print('and Y = %.1f' % Y +'%' + ' good representations if there is a representation.' )
    else:
        print('Dataset file does not exist, please check and run again.')
else:
    print('Please run with a dataset per: "python eval_wordEmbeddings.py dataset_filename" ')