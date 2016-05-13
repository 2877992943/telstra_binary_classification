#!/usr/bin/env python
# encoding=utf-8



"""
discrete fea count
"""


import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import cPickle
import pylab as plt


dataPath='/home/yr/telstra/'

def eval_wrapper(yhat, y):  #pred true
    y = np.array(y);print y[:10]
    y = y.astype(int);print yhat[:10]
    yhat = np.array(yhat)
    #yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)  
    #####accuracy
    #err=np.sum((y-yhat)*(y-yhat))/float(y.shape[0])
    #return err
    #######-loglikely
    return np.mean(-np.log(yhat+0.00001)*y-(1.-y)*np.log(1.-yhat+0.00001) )
    
    #return quadratic_weighted_kappa(yhat, y)
   



def get_params(maxDepth):
    
    plst={ 
  	"objective": 'multi:softprob',#"binary:logistic",
   	"booster": "gbtree",
   	"eval_metric": "auc",
  	"eta": 0.01, # 0.06, #0.01,
  	#"min_child_weight": 240,
	"silent":1,
   	"subsample": 0.75,
   	"colsample_bytree": 0.68,
   	"max_depth": maxDepth,
	"num_class":3
	}

    return plst


def pad(train):
	train.v22.fillna('',inplace=True)
	padded=train.v22.str.pad(4)
	spadded=sorted(np.unique(padded))
	v22_map={}
	c=0
	for i in spadded:
		v22_map[i]=c
		c+=1
	train.v22=padded.replace(v22_map,inplace=False)
	return train


def save2pickle(c,name):
    write_file=open(dataPath+str(name),'wb')
    cPickle.dump(c,write_file,-1)#[ (timestamp,[motion,x,y,z]),...]
    write_file.close()
 
def load_pickle(path_i):
    f=open(path_i,'rb')
    data=cPickle.load(f)#[ [time,[xyz],y] ,[],[]...]
    f.close()
    #print data.__len__(),data[0]
    return data	


def str2dummy(fea_xi,allFeaList,logVolumeDic_xi):#allFeaList [string,...]1921
	#print 'fea xi',len(fea_xi) #9 nonzero dimention->1920dim 0-1 vec
	vec01=[]
	#remove volume(int) in fea_xi,allFeaList
	#allFeaList=[f  for f in allFeaList if type(f)==str]
	#fea_xi=[f for f in fea_xi if type(f)==str ]
	#
	for fi in allFeaList:#for each string_fea in 1921 ,include: string int
		if  fi in fea_xi:#'log_feature'
			#print fi
			v=[logVolumeDic_xi[fi] if type(fi)==str and 'feature' in fi else 1][0]
			vec01.append(v)
		else:vec01.append(0)
	return np.array(vec01)
			

def get_logVolume(patch):
	dic={}
	log=np.unique(patch.log_feature.values)
	for logI in log:
		volumeI=np.unique(patch[patch.log_feature==logI].volume.values)[0]
		dic[logI]=volumeI
	return dic


def howMany(strFi_short,strList_xi):
	num=0
	for strI in strList_xi:
		if type(strI)==str and strFi_short in strI:
			num+=1;#print strI
	return num
		
		
def uniqueInorder(strArr) : #arr[str] ->list
	lis=[]
	for strI in strArr:
		if strI not in lis:lis.append(strI)
	return lis
			
def normal(xi_dic,normDic):#{'severity_type': 1, 'location': 1, 'event_type': 11, 'resource_type': 5, 'feature': 20}
	for k,v in normDic.items():
		while len(xi_dic[k])<v:
			xi_dic[k].append(-1)
	#xi={'event_type': ['34', '35', -1, -1, -1, -1, -1, -1, -1, -1, -1], 'feature': ['312', '232', -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 'severity_type': ['2'], 'volume': [11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1], 'location': ['460'], 'resource_type': ['2', -1, -1, -1, -1]}
	
	#get values ,transform into int
	
	return xi_dic
			

if __name__=='__main__':

	 
	 
	 

	#########
	#get 01fea 1900(all01 include location),790(all01 except for location),combine count fea
	#########
	 
	dataDicCount,targetDic,_=load_pickle(dataPath+'dataTargetFeaAll_count')
	#dataDicCount,targetDic,_=load_pickle(dataPath+'dataTargetFeaAll_count_feaVol')
	#dataDicCount,targetDic,_=load_pickle(dataPath+'dataTargetFeaAll_count_poly')
	#all 01 include location, notall01 
	dataDic_all01,dataDic_notall01,dataDic_notall01_noInt,targetDic,allFeaList=load_pickle(dataPath+'dataTarget_FeaAll01_notall01')

	timeArr,locFreq=load_pickle(dataPath+'timeInd_locFreq');
	timeLoc=np.concatenate((timeArr.reshape((-1,1)),locFreq.reshape((-1,1)) ),axis=1);print timeLoc.shape
	 

	dataDic_01_count={}
	ind=0
	for idi,arr in dataDic_notall01_noInt.items()[:]:
		 
		count=np.array(dataDicCount[idi]);#print count.shape,arr.shape,timeLoc[ind,:]
		#
		x=np.concatenate((count,arr,timeLoc[ind,:]))
		dataDic_01_count[idi]=x;#print x.shape
		#
		ind+=1
	save2pickle([dataDic_01_count,targetDic],'xy_dummy_Count_timeLoc')
		
	   

	""" 
	####see each xi ,at most how many event_type,
	  
	strFiDic={'event_type':0,'feature':0,'resource_type':0,'severity_type':0,'location':0}
	#{'severity_type': 1, 'location': 1, 'event_type': 11, 'resource_type': 5, 'feature': 20}
	dataDic,targetDic,allFeaList=load_pickle(dataPath+'dataTargetFeaAll_3')#{id:[strFea...]
	for idi,strList in dataDic.items()[:]:#each xi
		#print strList
		for strFi_short,mostNum in strFiDic.items():
			num=howMany(strFi_short,strList)
			if num>mostNum:strFiDic[strFi_short]=num
	print strFiDic
	"""
	 

	
			
			
				
	 
				
		
	
	 		
		
		
	
	


	"""
	#split discrete variable 'bc'
	print('Clearing...')
	# v22 v56 v125 'bcn'remain,add more variable,err not decrease
	 
	train['v22_0']=train.v22.str[0]; 
	train['v22_1']=train.v22.str[1]; 
	train['v22_2']=train.v22.str[2]; 
	train['v22_3']=train.v22.str[3]; 
	train['v56_0']=train.v56.str[0]; 
	train['v56_1']=train.v56.str[1];
	train['v125_0']=train.v125.str[0]; 
	train['v125_1']=train.v125.str[1];
	train['v113_0']=train.v113.str[0]
	train['v113_1']=train.v113.str[1]
	strList=['v22','v56','125','113']
	newfea=[]
	for strI in strList:
		for col in train.columns:
			if col.find(strI+'_')!=-1:
				print col
				serial=train[col].values
				print np.unique(serial).shape
				print np.unique(serial)[:50]
				#
				s, tmp_indexer = pd.factorize(train[col])
				print s.shape
				newfea.append(s)
	newfea=np.array(newfea).T#[d,n]	->[n,d]
	print newfea.shape#[n,10]
	save2pickle(newfea,'splitFea')
	
	
	


	#pad v22
	#train=pad(train)
	#

	"""




	""" 
	#dropna not factorized,see complete dataset without nan
	train1=train.dropna(axis=1,how='any')#12 fea with all value
	train2=train.dropna(axis=0,how='any');print 'complete data',train2.values.shape #complete fea data
	test2=test.dropna(axis=0,how='any')
	train2test2=np.concatenate((train2.values,test2.values),axis=0);print train2test2.shape#not factorized
	print 'all value fea',train1.columns
	test1=test[train1.columns]

	#train=train1;test=test1 

	
	
	#
	  
	# fill na ,factorize str feature
	missFea=[];completeFea=[]
	feaInd=-1
	for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems())[:]:
	    feaInd+=1
	    # each columns,fea
	    valuePercnt_train=train[train_name].count()/float(train.values.shape[0])
	    valuePercnt_test=test[test_name].count()/float(test.values.shape[0])
	    #print 'non-nan value fea',train_name,train_series.dtype,valuePercnt_train,valuePercnt_test
	    ##
	    if train_series.dtype == 'O':
		#for objects: factorize
		
		
		train[train_name], tmp_indexer = pd.factorize(train[train_name]);
		#print np.unique(tmp_indexer).shape
		test[test_name] = tmp_indexer.get_indexer(test[test_name])
		if valuePercnt_test+valuePercnt_train<2.:missFea.append(feaInd)
		else:completeFea.append(feaInd)
		
		 
		#but now we have -1 values (NaN)
	    else:
		#print train_name,np.unique(train_series).shape
		 
		#for int or float: fill NaN with mean
		if valuePercnt_test+valuePercnt_train<2.:
			missFea.append(feaInd)
			tmp_len = len(train[train_series.isnull()]); 
			if tmp_len>0:
		    		train.loc[train_series.isnull(), train_name] = -1000
			#and Test
			tmp_len = len(test[test_series.isnull()])
			if tmp_len>0:
		    		test.loc[test_series.isnull(), test_name] = -1000   

			
		else:
			completeFea.append(feaInd)
			tmp_len = len(train[train_series.isnull()]); 
			if tmp_len>0:
		    		train.loc[train_series.isnull(), train_name] = train_series.mean()
			#and Test
			tmp_len = len(test[test_series.isnull()])
			if tmp_len>0:
		    		test.loc[test_series.isnull(), test_name] = train_series.mean()  #TODO

	"""



	"""
	print len(missFea),len(completeFea)
	##
	missInd=list(np.where(train.values==-1)[0])+list(np.where(train.values==-1000)[0])
	train1=train.drop(missInd,axis=0,inplace=False)
	missInd=list(np.where(test.values==-1)[0])+list(np.where(test.values==-1000)[0])
	test1=test.drop(missInd,axis=0,inplace=False)
	train2test2=np.concatenate((train1,test1),axis=0);print 'complete data',train2test2.shape
	save2pickle([missFea,completeFea,train.values,test.values,train2test2],'midData')
	"""  
	 


	"""


	 
	 
	#####################
	#xgboost
	###################
	# convert data to xgb data structure
	missing_indicator=-1000
	xgtrain = xgb.DMatrix(train.values, target.values,missing=missing_indicator);
	
	#xgtest = xgb.DMatrix(test,missing=missing_indicator)
 
	 

	 
	 
	# train model
	print('Fit different model...')
	for boost_round in [50,100][:1]:
		
		 
		for maxDepth in [7,14][:1]:#7  14
			xgboost_params = get_params(maxDepth)
			 
			# train model
			 
			
			#clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)
			clf=xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round)

			# train error
			train_preds = clf.predict(xgtrain, ntree_limit=clf.best_iteration)
			print maxDepth,boost_round
			print('Train err is:', eval_wrapper(train_preds, target.values))# 50 7 0.19
			 
	 		 
	 
	"""
	



	""" 
	#test predict
	print('Predict...')
	test_preds = clf.predict(xgtest, ntree_limit=clf.best_iteration)
	# Save results
	#
	preds_out = pd.DataFrame({"ID": ids, "PredictedProb": test_preds})
	preds_out.to_csv("../acc_process_submission.csv")
	#
	"""
	 
	 
	

	
	
	  
		
		
	
	
	

 



"""
{id:{event:[11 events at most for one xi] in order---------------53 kinds
     feature:[20] -----------------386
	resource:[5]------------------10
	severity:[1]------------------------5
	location:[1]-------------------------------929+1039
	volume:[20]------------------------341
"""	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



