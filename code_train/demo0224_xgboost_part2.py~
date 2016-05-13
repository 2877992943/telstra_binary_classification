#!/usr/bin/env python
# encoding=utf-8




import pandas as pd
import numpy as np
import xgboost as xgb
from scipy.optimize import fmin_powell
from ml_metrics import quadratic_weighted_kappa
import cPickle
import pylab as plt


dataPath='/home/yr/telstra/'

def eval_wrapper(ypred, ytrue):  #pred true
    #make sure 012
    #ypred=np.concatenate((ypred,np.array([0,1,2]) )) ypred is [0.1 0.2 0.7]
    ytrue=np.concatenate((ytrue,np.array([0,1,2]) ))
    print ypred.shape,ytrue.shape
    if len(ytrue.shape)!=2:
	dmmat=pd.get_dummies(np.array(ytrue))
    	ytrue=dmmat.values #[n,3]
    if len(ypred.shape)!=2:
	dmmat=pd.get_dummies(np.array(ypred))
    	ypred=dmmat.values #[n,3]
    ytrue=ytrue[:-3,:]


    
    
    #y = np.array(y);print y[:10]
    #y = y.astype(int);print yhat[:10]
    #yhat = np.array(yhat)
    #yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)  
    #####accuracy
    #err=np.sum((y-yhat)*(y-yhat))/float(y.shape[0])
    #return err
    #######-loglikely
    return -np.mean( np.sum(ytrue*np.log(ypred+0.00001), axis=1) )#[n,3]->[n,]->1x1
	
    
    #return quadratic_weighted_kappa(yhat, y)


def get_params(maxDepth):
    
    plst={ 
  	"objective": 'multi:softprob',#"binary:logistic",
   	"booster": "gbtree",
   	"eval_metric": "auc",
  	"eta": 0.01, # 0.06, #0.01,
  	#"min_child_weight": 20,
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

def calcDist(dataC,xi,xiObsInd,xiMissInd):
	n=dataC.shape[0]
	xObs=xi[xiObsInd]
	xMiss=xi[xiMissInd]
	dist=np.tile(xObs.reshape((1,-1)),(n,1)) - dataC[:,xiObsInd]#[n,d]
	dist=np.sum(dist*dist,axis=1)
	
	distInd=np.argsort(dist)[:10]
	return distInd



def vote(vec):
	dic={}
	for v in vec:
		if v not in dic:
			dic[v]=1
		else:dic[v]+=1
	ll=sorted(dic.iteritems(),key=lambda asd:asd[1],reverse=False)
	#print dic
	return ll[-1][0]



def pca(x):
	x=x.T#[n,131]->[131,n]
	m=np.mean(x,axis=1);print 'mean',m.shape #[131,]
	removeMean=x-np.tile(m.reshape((-1,1)),(1,x.shape[1]))  #[131,1]->[131,n]
	cov=np.dot(removeMean,removeMean.T);print 'cov',cov.shape
	Sigma,U=np.linalg.eig(np.mat(cov));print 'u sigma',U.shape,Sigma.shape#[d,d][d,]
	SigmaInd=np.argsort(Sigma)[:-100000:-1];print SigmaInd.shape
	Sigma=Sigma[SigmaInd]
	U=U[:,SigmaInd]

	#energy of sigma
	Sigma2=Sigma*Sigma
	energy=0;tt=np.dot(Sigma,Sigma)
	for i in range(Sigma.shape[0]):
		energy+=Sigma2[i]
		#print energy/tt
		if energy/tt>=0.99:
			print '0.99',i
			break

	##
	U=U[:,:i]#[d,30]
	xrot=U.T*removeMean #[30,d][d,n] ->[30,n]
	xrot=xrot.T #[n,30]
	return xrot


 
 
	 
	
		
	

if __name__=='__main__':
	 
  	 
	#dataDic,targetDic,feaAllList=load_pickle(dataPath+'dataTargetFeaAll_2') #{id:fea01...} {id:y...} [fea...]
	#dataDic,targetDic=load_pickle(dataPath+'xy_dummy_Count_seve1')#notall01_count
	#dim=len(feaAllList);print 'dim',dim,dataDic.values()[0].shape
	#dataDic,targetDic=load_pickle(dataPath+'xy_dummy_Count_resource8')
	#dataDic,targetDic0=load_pickle(dataPath+'xy_dummy_Count_event11');print len(dataDic)
	dataDic,targetDic0=load_pickle(dataPath+'xy_dummy_Count_chooseFea')
	#targetDic1=load_pickle(dataPath+'targetDic_half')
	#
	part1 = pd.read_csv("../telstra_submission0229_part1.csv");
	part1Ind=set(list(part1['id'].values))#test part1 seve2_resource2
	part2Ind=set(targetDic0.keys())-part1Ind;print len(part1Ind),len(targetDic0.keys())
	##
	#split train set test set
	trainList=[];testList=[];testId=[];targetList=[]
	 
	#for idi,feaList in dataDic.items():
	for idi in part2Ind:
		feaList=dataDic[idi]#arr
		##more trainset 
		if targetDic0[idi] in [0,1,2]:# with label !=Nan
			trainList.append(feaList)
			targetList.append(targetDic0[idi])
		###original testset
		if targetDic0[idi] not in [0,1,2]:
			testList.append(feaList)
			testId.append(idi)
	
	 
	
	 
	print 'y,test,whole',len(targetList),len(testId),len(targetDic0)
	train=np.array(trainList)
	test=np.array(testList)
	target=np.array(targetList)
	print 'train test y',train.shape,test.shape,target.shape
	save2pickle([train,test,target,testId],'trainTestArr_target_testId_2')
	 
	train,test,target,testId=load_pickle(dataPath+'trainTestArr_target_testId_2')
	 
	 
			
		

	  
	#####################
	#xgboost
	###################
	# convert data to xgb data structure
	from sklearn.cross_validation import train_test_split
	missing_indicator=-1
	xtrain,xtest,ytrain,ytest=train_test_split(train,target,test_size=0.1)
	xgtrain = xgb.DMatrix(xtrain, ytrain,missing=missing_indicator);
	print 'x y',xtrain.shape,ytrain.shape
	xgtest = xgb.DMatrix(xtest,missing=missing_indicator)
	
	xgtest1 = xgb.DMatrix(test)# no label
 
	 

	 	
	  
	# train model
	print('Fit different model...')
	for boost_round in [500,1000][:1]:
		
		 
		for maxDepth in [14,32][:1]:#7  14
			xgboost_params = get_params(maxDepth)
			 
			# train model
			 
			
			#clf = xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round,verbose_eval=True,maximize=False)
			clf=xgb.train(xgboost_params,xgtrain,num_boost_round=boost_round)

			# train test  error
			print 'train test error...'
			train_preds = clf.predict(xgtrain, ntree_limit=clf.best_iteration);print train_preds.shape
			test_preds=clf.predict(xgtest,ntree_limit=clf.best_iteration);print test_preds.shape
			print 'depth round',maxDepth,boost_round
			print('Train err is:', eval_wrapper(train_preds, ytrain))# 
			print('Test err is(see whether overfit:', eval_wrapper(test_preds, ytest))

			#
			#save2pickle(clf,'clf_'+str(maxDepth) )
	 

	   
				
				
	 		
			
		



    
	################
	#test predict
	
	 
	 
	#clf=load_pickle(dataPath+'clf_'+str(maxDepth) )
	print('Predict...')
	test_preds = clf.predict(xgtest1, ntree_limit=clf.best_iteration)#[n,3]
	# Save results
	#
	#preds_out = pd.DataFrame({"ID": ids, "PredictedProb": test_preds})
	preds_out = pd.DataFrame({'id':testId,"predict_0": test_preds[:,0],"predict_1": test_preds[:,1],\
						"predict_2": test_preds[:,2]})
	preds_out.to_csv("../telstra_submission0229_part2.csv")
	  
	 	
	  
	
	 


	 


#####
#pca continuVariable  loss 0.5899
#continuVarible loss 0.5813	 
	 
##not pca continuVariable only trainLoss 0.5789  testLoss0.5841	
##pca continuVariable only     trainLoss 0.5891  testLoss 0.5928
#polynomial continuVariable
# dummy(discreteVariable)_notPca + continuVariable_pca  trainloss 0.5780  testloss 0.5817

	
	
	  
		
		
	
	
	

 	
 

	 




	 

	 
	 
	 


 	
		
    
 
	
		
	
   		 



