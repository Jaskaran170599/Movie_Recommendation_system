import pandas as pd
import pickle as pkl
from sklearn.metrics.pairwise import cosine_similarity
import bert
import numpy as np
from bert.extract_features import *

movie_data=pd.read_csv("./movies_metadata.csv")

with open("./emb.pickle","rb") as file:
    emb=pkl.load(file)                        #loading overview embeddings

def make_dataset(data):                       # extracting data_info
    data['tagline']=data["tagline"].fillna(" ")
    data=data.loc[:,["original_title","genres","overview"]]
    print(data.shape)
    data=data.dropna()
    print(data.shape)
    return data.loc[:,["original_title","genres","overview"]]

data_info=make_dataset(movie_data)
titles=data_info["original_title"].apply(lambda x:"".join(x.lower().split()))

def get_similar_titles(title):     #return similar titles
    index,j=[],0
    title="".join(title.lower().split())
    for i in titles:
        if title in i:
            index.append(j)
        j+=1
    return data_info.iloc[index]["original_title"],index

def get_nearest_movie(title,n=10):    #return 10 similar movies based on cosine similarity 
    rec,rec_movies=[],[]
    rec_movies.append((title,data_info[data_info["original_title"]==title]["overview"]))
    vec=emb[title]
    for i in emb.keys():
        if i == title:
            continue
        x=cosine_similarity(vec.reshape(1,-1),emb[i].reshape(1,-1))
        rec.append((x,i))
    for j,i in sorted(rec,key=lambda x : x[0],reverse=True)[:n]:
        rec_movies.append((i,data_info[data_info["original_title"]==i]["overview"]))
    return rec_movies  

#to extract embeddings from overview 

tokenizer = bert.tokenization.FullTokenizer(
  vocab_file="./tmp/wwm_uncased_L-24_H-1024_A-16/vocab.txt", do_lower_case=True)

layer_indexes = [int(x) for x in [-1,-2,-3,-4]]              #last 4 hidden layers

bert_config = bert.modeling.BertConfig.from_json_file("./tmp/wwm_uncased_L-24_H-1024_A-16/bert_config.json")#model configs

tokenizer = bert.tokenization.FullTokenizer(
  vocab_file="./tmp/wwm_uncased_L-24_H-1024_A-16/vocab.txt", do_lower_case=True)

is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2    #
run_config = tf.contrib.tpu.RunConfig(                          #
  master=None,                                                  #    Setting up configurations
  tpu_config=tf.contrib.tpu.TPUConfig(                          #  
      num_shards=None,                                          #
      per_host_input_for_training=is_per_host))                 #   

model_fn = model_fn_builder(                                        #
  bert_config=bert_config,                                          #
  init_checkpoint="./tmp/wwm_uncased_L-24_H-1024_A-16/bert_model.ckpt",  #  funtion to build pretrained model
  layer_indexes=layer_indexes,                                      #
  use_tpu=False,                                                    # 
  use_one_hot_embeddings=False)                                     #

estimator = tf.contrib.tpu.TPUEstimator(                            #
  use_tpu=False,                                                    #
  model_fn=model_fn,                                                # predictor to extract features 
  config=run_config,                                                #
  predict_batch_size=12)   

def extract_feat(text):  #extract embeddings
    
    text_=[InputExample(unique_id=0, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = text, 
                                                                   text_b = None)]
    train_data=convert_examples_to_features(text_
                                        ,256,tokenizer,) #converting to features expected by the model
    input_fn = input_fn_builder(features=train_data, seq_length=256)   #prepares input
    
    for result in estimator.predict(input_fn, yield_single_examples=True):
        l1=np.mean(axis=0,a=result['layer_output_0'])     #
        l2=np.mean(axis=0,a=result['layer_output_1'])     # Last 4 hidden layers features
        l3=np.mean(axis=0,a=result['layer_output_2'])     # 
        l4=np.mean(axis=0,a=result['layer_output_3'])     #

        emb=np.mean(axis=0,a=[l1,l2,l3,l4])               # taking mean to get sentence embeddings
        return emb                                                  # saving with movie titles
    return -1      

def get_nearest_overview(text,n=10):   #return 10 similar movies based on given overview
    vec=extract_feat(text)
    rec_movies,rec=[],[]
    for i in emb.keys():
        x=cosine_similarity(vec.reshape(1,-1),emb[i].reshape(1,-1))
        rec.append((x,i))
    for j,i in sorted(rec,key=lambda x : x[0],reverse=True)[:n]:
        rec_movies.append((i,data_info[data_info["original_title"]==i]["overview"],j))
    return rec_movies 