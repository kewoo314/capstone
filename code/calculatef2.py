#!/usr/bin/env python3

import json 
import sys
import Id
import sklearn.model_selection
import collections
import numpy as np
import tensorflow as tf
import random
import functools
import itertools

def empty_vul_ben_dict():
    return {"vul":0, "ben":0}

def create_vul_percent(dataset, data_labels, rounding):
    
    try:
        rounding = int(rounding)
    except:
        rounding = 10

    words = collections.defaultdict(empty_vul_ben_dict)
    for index, terms in enumerate(dataset):
        for word in terms:
            if data_labels[index] == 1:
                words[word]["vul"]+=1
            else:
                words[word]["ben"]+=1

    vul_dict = {word: round(words[word]["vul"] / (words[word]["vul"]+words[word]["ben"]),
                rounding) for
            word in words }
    

    return vul_dict

def make_calc(data, vul_dict):
    data_set = []

    vul_avg = sum(vul_dict.values())/len(vul_dict.values())

    for func in data:
        vul_array = [ vul_dict[word] if word in vul_dict else vul_avg for word
                in func ]
        count = len(vul_array)
        if count == 0:
            data_set.append([0]*4)
        else:
            data_set.append([count,max(vul_array),min(vul_array),sum(vul_array)/count])

    return data_set

def make_ratio(data, vul_dict, n=10):
    data_set = []

    vul_avg = sum(vul_dict.values())/len(vul_dict.values())
    
    for func in data:
        vul_array = [ vul_dict[word] if word in vul_dict else vul_avg for word
                in func ]
        vul_array.sort(reverse=True)
        vul_array.extend([vul_avg]*n)
        vul_array=vul_array[:n]
        data_set.append(vul_array)


    return np.array(data_set)

def make_str(data):
    data_words = []

    
    for func in data:

        clean_call = [ word for word in func if word is not None ]
        if len(clean_call) == 0:
            data_words.append(" ")
        else:
            data_words.append(" ".join(clean_call))


    return np.array(data_words)




def embed_model(vocab):

    vocab = np.array(vocab)

    v_layer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=None,
            output_mode='int', output_sequence_length=20)#10)
    v_layer.adapt(vocab)
    num_vocab = len(v_layer.get_vocabulary())
    
    start = tf.keras.layers.Input(shape = vocab.shape[1:],dtype="string")
    layer = v_layer(start)
    layer = tf.keras.layers.Embedding(num_vocab, 2, mask_zero=True,
            input_length=20)(layer)#10)(layer)
    layer = tf.keras.layers.Dense(128, activation='relu')(layer)
    end = tf.keras.layers.Flatten()(layer)

    return start, end

def dense_model(data):

    data = np.array(data)

    start = tf.keras.layers.Input(shape = data.shape[1:])
    layer = start
    layer = tf.keras.layers.Dense(128, activation='relu')(layer)
    end = tf.keras.layers.Flatten()(layer)

    return start, end

def multi_model( starts, ends ):
    
    layer = tf.keras.layers.concatenate(ends)
    layer = tf.keras.layers.Dense(128)(layer)
    final = tf.keras.layers.Dense(1,activation="sigmoid")(layer)

    return tf.keras.Model(inputs=starts, outputs=[final])

def old_model(data):
    
    start = tf.keras.layers.Input(shape = data.shape[1:])
    layer = start
    layer = tf.keras.layers.Dense(256, activation='relu')(layer)
    layer = tf.keras.layers.Flatten()(layer)
    end = tf.keras.layers.Dense(1,activation="sigmoid")(layer)
    
    return tf.keras.Model(inputs=[start],outputs=[end])

def shuffle(data, labels):

    merge = list(zip(data,labels))
    random.shuffle(merge)
    data, labels = zip(*merge)
    return np.array(data), np.array(labels)

def balance(data_set, labels):
    return data_set, labels

def build(train, train_labels, test, test_labels, datatype, process, modifier):
    style_mod = None
    test_style_mod = None
    model = None

    base_train = process_data(train, datatype, process)
    base_test = process_data(test, datatype, process)

    if "ratio" in modifier:
        vul_dict = create_vul_percent(base_train, train_labels, modifier[-1])
        style_mod = make_ratio(base_train, vul_dict)
        test_style_mod = make_ratio(base_test, vul_dict)
        model = dense_model(style_mod)

    elif modifier == "calc":
        vul_dict = create_vul_percent(base_train, train_labels, modifier[-1])
        style_mod = make_calc(base_train, vul_dict)
        test_style_mod = make_calc(base_test, vul_dict)
        model = dense_model(style_mod)

    elif modifier == "str":
        style_mod = make_str(base_train)
        test_style_mod = make_str(base_test)
        model = embed_model(style_mod)

    return np.array(style_mod), np.array(test_style_mod), model
    
def ngramize(data, num, granularity):
    num=int(num)
    final_array = []
    if granularity == "C":
        for word in data:
            if word is None:
                continue
            for start in range(len(word)-num+1):
                final_array.append(word[start:start+num])
    elif granularity == "I":
        final_array=["-".join(combo) for combo in
                itertools.combinations(data,num) if not None in combo]
    elif granularity == 'T':
        subarray = []
        for word in data:
            if word is not None:
                subarray.extend(Id.split(word))
        final_array=["-".join(combo) for combo in
                itertools.combinations(subarray,num) if not None in combo]
        

    return final_array

def process_data(data,datatype,process):
    final_array = []
    for d in data:
        if process == "raw":
            final_array.append(d[datatype])
        elif process == "terms":
            subarray = []
            for word in d[datatype]:
                if word is not None:
                    subarray.extend(Id.split(word))
            final_array.append(subarray)
        elif "gram" in process:
            final_array.append(ngramize(d[datatype], process[0], process[-1]))

    return final_array

def get_prog(filename):
    last_slash = filename.rfind("/")
    if last_slash != -1:
        filename = filename[last_slash+1:]
    vul_pos = filename.rfind("_non_vulnerable")
    if vul_pos == -1:
        vul_pos = filename.rfind("_vulnerable")
    return filename[:vul_pos]


in_files_args = True
all_files = []
run = []

num_splits = int(sys.argv[1])
num_rep = int(sys.argv[2])
outfile = sys.argv[3] if sys.argv[3] != "None" else False

for arg in sys.argv[4:]:
    
    if arg == '-o':
        in_files_args = False
    elif in_files_args:
        all_files.append(arg)
    else:
        data, process, transform = arg.split('_')
        run.append((data, process, transform))

functions = []
labels = []
index=0
all_funcs={}
for name_file in all_files:
    name_dict = json.load(open(name_file))
    prog = get_prog(name_file)
    for name,props in name_dict.items():
        fullname = prog+name
        if fullname in all_funcs:
            labels[all_funcs[fullname]]=1
            continue
        all_funcs[fullname]=index
        index+=1
        functions.append({"fn":[name],
            "cn":props["called"], "fp": props["formal_names"],
            "lv":props["locals"], "prog":get_prog(name_file)})
        labels.append(0 if "non_vulnerable" in name_file else 1)

functions = np.array(functions)
labels = np.array(labels)

all_f2 = []

if outfile:
    predications = open(outfile,"a")

kf = sklearn.model_selection.RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_rep, random_state = 529)
count=0
for train_index,test_index in kf.split(functions, labels): 
    
    train = functions[train_index]
    train_labels = labels[train_index]

    train, train_labels = shuffle(train, train_labels)

    test = functions[test_index]
    test_labels = labels[test_index]


    #train, train_labels = balance(train, train_labels)
    prebuilt = functools.partial(build, train, train_labels, test, test_labels)


    inputs = []
    outputs = []
    models = []
    names = []
    src = []
    split = []
    mod = []
    for pair in run:
        i, o, m = prebuilt(*pair)
        #print(pair,i,o)
        inputs.append(i)
        outputs.append(o)
        models.append(m)
        names.append("_".join(pair))
        src.append(pair[0])
        split.append(pair[1])
        mod.append(pair[2])

    model = multi_model(*zip(*models))

    class_weight = {0:1.0, 1:4.0}

    model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy',tf.keras.metrics.Precision(name="prec"),tf.keras.metrics.Recall(name="recall")])
    model.fit(inputs,train_labels,class_weight=class_weight, verbose=0, epochs = 20)#, validation_split=0.2)

    if outfile:
        each_guess = model.predict(outputs,verbose=0)
 
        for i in range(len(each_guess)):
            print(f"{count:02d}",test[i]["prog"]+"-"+"-".join(test[i]["fn"]),"-".join([ "_".join(x) for x in run]),each_guess[i][0],test_labels[i],sep=",",file=predications)

 
    count+=1

    loss,acc,prec,recall = model.evaluate(outputs, test_labels, verbose=0)
    if 4*prec+recall == 0:
        f2 = 0
        f1 = 0
    else:
        f2 = 5 * prec * recall / (4* prec + recall)
        f1 = 2 * prec * recall / (prec + recall)
    print("-".join(src),"-".join(split),"-".join(mod),round(f2,3),round(f1,3), round(prec,3), round(recall,3),sep=",")
#    all_f2.append(round(f2,3))

all_f2.sort()
#print("-".join(names),
#        all_f2[int(0.05*len(all_f2)-1)],round(sum(all_f2)/len(all_f2),3),all_f2[int(0.95*len(all_f2)-1)], sep=",")
