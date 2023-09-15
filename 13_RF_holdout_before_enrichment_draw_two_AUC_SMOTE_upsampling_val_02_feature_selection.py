import sys,os,argparse
import pandas as pd
import numpy as np
from sklearn import datasets
from scipy import stats as stats
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score 
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import joblib
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance
# https://kiwidamien.github.io/how-to-do-cross-validation-when-upsampling-data.html

def warn(*args, **kwargs):
	pass
import warnings
warnings.warn = warn

def get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds, num_pos_class, num_neg_class):
	tp = tpr * num_pos_class
	tn = (1 - fpr) * num_neg_class
	acc = (tp + tn) / (num_pos_class + num_neg_class)
	best_threshold = thresholds[np.argmax(acc)]
	return(best_threshold)

def get_metric_and_best_threshold_from_roc_curve_best_measure(y,y_prob,measure):
	max_f1 = -1
	optimal_thresh = ''
	for thr in np.arange(0.01, 1, 0.01):
		thr_pred = y_prob[1].copy()
		thr_pred[thr_pred >= thr] = 1
		thr_pred[thr_pred < thr] = 0
		# Eliminates cases where all predictions are negative and
		if sum(thr_pred) > 1:
			if (measure == 'f1'):
				f1 = f1_score(y, thr_pred, pos_label=1)
			elif (measure == 'accuracy'):
				f1 = accuracy_score(y, thr_pred)
			elif measure == 'auprc':
				f1 = average_precision_score(y, thr_pred)
			else:
				print('%s is not a scoring option for model thresholding' % measure)
				exit()
			if f1 > max_f1:
				max_f1 = f1
				optimal_thresh = thr
	return(optimal_thresh)

def DefineClf_RandomForest(n_estimators,max_depth,max_features):
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(n_estimators=int(n_estimators), 
		max_depth=max_depth, 
		max_features=max_features,
		criterion='entropy',
		random_state=42)
	return clf

# determine the direction of importance
def Imp_dir(X, y, imp):
	for i in range(0,imp.shape[0]):
		feature = imp.iloc[i,0]
		pcc = np.corrcoef(np.array(X[feature]), np.array(y))[0,1]
		if pcc < 0:
			imp.iloc[i,1] = -1 * imp.iloc[i,1]
	return (imp)

def score_model_SMOTE(model, cv_num, X, y):
	from imblearn.over_sampling import SMOTE
	cv = StratifiedKFold(n_splits=cv_num, random_state=42,shuffle=True)
	smoter = SMOTE(sampling_strategy='not majority',random_state=42,k_neighbors=3)
	scores = {}
	scores['roc_auc'] = []
	scores['f1'] = []
	scores['accuracy'] = []
	cv_proba = np.empty((0,2))
	y_upsample = []
	for train_fold_index, val_fold_index in cv.split(X, y):
		# Get the training data
		X_train_fold, y_train_fold = X.iloc[train_fold_index], y[train_fold_index]
		# Get the validation data
		X_val_fold, y_val_fold = X.iloc[val_fold_index], y[val_fold_index]
		# Upsample only the data in the training section
		X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,y_train_fold)
		# Upsample only the data in the validation section
		X_val_fold_upsample, y_val_fold_upsample = smoter.fit_resample(X_val_fold,y_val_fold)
		# Fit the model on the upsampled training data
		model_obj = model.fit(X_train_fold_upsample, y_train_fold_upsample)
		# Score the model on the (non-upsampled) validation data
		scores['roc_auc'].append(roc_auc_score(y_val_fold_upsample, model_obj.predict(X_val_fold_upsample)))
		scores['f1'].append(f1_score(y_val_fold_upsample, model_obj.predict(X_val_fold_upsample)))
		scores['accuracy'].append(accuracy_score(y_val_fold_upsample, model_obj.predict(X_val_fold_upsample)))
		y_upsample = np.concatenate((y_upsample,y_val_fold_upsample),axis=0)
		proba = model_obj.predict_proba(X_val_fold_upsample)
		cv_proba = np.concatenate((cv_proba,proba),axis=0)
	return(scores,y_upsample,cv_proba)

def IMP_permutation_SMOTE(model, cv_num, X, y, score):
	from imblearn.over_sampling import SMOTE
	cv = StratifiedKFold(n_splits=cv_num, random_state=42,shuffle=True)
	smoter = SMOTE(sampling_strategy='not majority',random_state=42,k_neighbors=3)
	imp_per = pd.DataFrame(index=X.columns, columns=['0'])
	for train_fold_index, val_fold_index in cv.split(X, y):
		# Get the training data
		X_train_fold, y_train_fold = X.iloc[train_fold_index], y[train_fold_index]
		# Get the validation data
		X_val_fold, y_val_fold = X.iloc[val_fold_index], y[val_fold_index]
		# Upsample only the data in the training section
		X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,y_train_fold)
		# Upsample only the data in the validation section
		X_val_fold_upsample, y_val_fold_upsample = smoter.fit_resample(X_val_fold,y_val_fold)
		# Fit the model on the upsampled training data
		model_obj = model.fit(X_train_fold_upsample, y_train_fold_upsample)
		# Score the model on the (non-upsampled) validation data
		imp_permutation = permutation_importance(model_obj, X_val_fold_upsample, y_val_fold_upsample, scoring=score, n_repeats=5, n_jobs=10, random_state=42)
		imp_permutation_2 = pd.DataFrame(imp_permutation.importances_mean)
		imp_permutation_2.index = X.columns		
		imp_per = pd.concat([imp_per,imp_permutation_2],axis=1)
	return(imp_per.iloc[:,1:])
	
def IMP_permutation(model, cv_num, X, y, score):
	from imblearn.over_sampling import SMOTE
	cv = StratifiedKFold(n_splits=cv_num, random_state=42,shuffle=True)
	imp_per = pd.DataFrame(index=X.columns, columns=['0'])
	for train_fold_index, val_fold_index in cv.split(X, y):
		# Get the training data
		X_train_fold, y_train_fold = X.iloc[train_fold_index], y[train_fold_index]
		# Get the validation data
		X_val_fold, y_val_fold = X.iloc[val_fold_index], y[val_fold_index]
		# Fit the model on the upsampled training data
		model_obj = model.fit(X_train_fold, y_train_fold)
		# Score the model on the (non-upsampled) validation data
		imp_permutation = permutation_importance(model_obj, X_val_fold, y_val_fold, scoring=score, n_repeats=5, n_jobs=10, random_state=42)
		imp_permutation_2 = pd.DataFrame(imp_permutation.importances_mean)
		imp_permutation_2.index = X.columns		
		imp_per = pd.concat([imp_per,imp_permutation_2],axis=1)
	return(imp_per.iloc[:,1:])

def cv_prediction_SMOTE(model, cv_num, X, y):
	from imblearn.over_sampling import SMOTE
	cv = StratifiedKFold(n_splits=cv_num, random_state=42,shuffle=True)
	smoter = SMOTE(sampling_strategy='not majority',random_state=42,k_neighbors=3)
	cv_proba = np.empty((0,2))
	y_upsample = []
	for train_fold_index, val_fold_index in cv.split(X, y):
		# Get the training data
		X_train_fold, y_train_fold = X.iloc[train_fold_index], y[train_fold_index]
		# Get the validation data
		X_val_fold, y_val_fold = X.iloc[val_fold_index], y[val_fold_index]
		# Upsample only the data in the training section
		X_train_fold_upsample, y_train_fold_upsample = smoter.fit_resample(X_train_fold,y_train_fold)
		# Upsample only the data in the validation section
		X_val_fold_upsample, y_val_fold_upsample = smoter.fit_resample(X_val_fold,y_val_fold)
		# Fit the model on the upsampled training data
		model_obj = model.fit(X_train_fold_upsample, y_train_fold_upsample)
		# Score the model on the (non-upsampled) validation data
		pred = model_obj.predict_proba(X_val_fold_upsample)
		y_upsample = np.concatenate((y_upsample,y_val_fold_upsample),axis=0)
		cv_proba = np.concatenate((cv_proba,pred),axis=0)
	return(y_upsample,cv_proba)


def main():
	parser = argparse.ArgumentParser(description='This code contains the RF model building. ')
	# Required
	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-file', help='matrix file used to train the model', required=True)
	# optional
	inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
	inp_group.add_argument('-smote', help='up-sampling the minority class or not, y or n',default='n')
	inp_group.add_argument('-cv_num', help='number of folds for cross validation',default=5)
	inp_group.add_argument('-score', help='score used to select final model',default='roc_auc')
	inp_group.add_argument('-feat', help='files with selected features',default = 'all')

	if len(sys.argv)==1:
		parser.print_help()
		sys.exit(0)
	args = parser.parse_args()	

	df = pd.read_csv(args.file, sep='\t',header=0,index_col=0)
	if args.feat != 'all':
		print('Using subset of features from: %s' % args.feat)
		with open(args.feat) as f:
			features = f.read().strip().splitlines()
			features = ['Class'] + features
		df = df.loc[:, features]

	X = df.drop('Class',axis=1)
	y = df.Class

	param_grid = {'max_depth':[3, 5, 10], 'max_features': [0.1, 0.5, 'sqrt', 'log2', None], 'n_estimators': [10, 100,500,1000]}
	if args.smote == 'y':
		from imblearn.over_sampling import SMOTE
		from sklearn.pipeline import Pipeline, make_pipeline
		from imblearn.pipeline import Pipeline, make_pipeline
		from sklearn.model_selection import KFold
		GS = {}
		AUCROC = {}
		F1_score = {}
		Accuracy_score = {}
		Proba = {}
		Imp_per = {}
		for n_estimators in param_grid['n_estimators']:
			for max_depth in param_grid['max_depth']:
				for max_features in param_grid['max_features']:
					clf = DefineClf_RandomForest(n_estimators,max_depth,max_features)
					CV_result = score_model_SMOTE(clf, cv_num=5, X=X, y=y)
					scores = CV_result[0]
					GS['%s__%s__%s'%(n_estimators,max_depth,max_features)] = np.array(scores[args.score]).mean()
					AUCROC['%s__%s__%s'%(n_estimators,max_depth,max_features)] = np.array(scores['roc_auc']).mean()
					F1_score['%s__%s__%s'%(n_estimators,max_depth,max_features)] = np.array(scores['f1']).mean()
					Accuracy_score['%s__%s__%s'%(n_estimators,max_depth,max_features)] = np.array(scores['accuracy']).mean()
					Proba['%s__%s__%s'%(n_estimators,max_depth,max_features)] = {}
					Proba['%s__%s__%s'%(n_estimators,max_depth,max_features)]['y_upsample_CV'] = CV_result[1]
					Proba['%s__%s__%s'%(n_estimators,max_depth,max_features)]['cv_proba'] = CV_result[2]
		parameter = max(GS, key=GS.get)
		n_estimators = int(parameter.split('__')[0])
		max_depth = int(parameter.split('__')[1])
		max_features = parameter.split('__')[2]
		if max_features == '0.1' or max_features == '0.5':
			max_features = float(max_features)
		if max_features == 'None':
			max_features = None
		Best_parameter = {'n_estimators':n_estimators,'max_depth':max_depth,'max_features':max_features}
		Best_score = GS[parameter]
		smoter = SMOTE(sampling_strategy='not majority',random_state=42,k_neighbors=3)
		y_upsample_CV = Proba[parameter]['y_upsample_CV']
		cv_proba = Proba[parameter]['cv_proba']
		#y_upsample_CV, cv_proba = cv_prediction_SMOTE(DefineClf_RandomForest(n_estimators,max_depth,max_features), int(args.cv_num), X, y)
		X_upsample, y_upsample = smoter.fit_resample(X,y)
		clf = DefineClf_RandomForest(n_estimators,max_depth,max_features)
		Imp_per = IMP_permutation_SMOTE(clf, cv_num=5, X=X, y=y, score = args.score)
		Final_model = DefineClf_RandomForest(n_estimators,max_depth,max_features).fit(X_upsample,y_upsample)
		#cv_proba = cross_val_predict(estimator=Final_model, X=X_upsample, y=y_upsample, cv=int(args.cv_num), method='predict_proba')

	else:
		grid_search = GridSearchCV(RandomForestClassifier(random_state=42,criterion="entropy"), param_grid, cv=int(args.cv_num), scoring=args.score, n_jobs=5)
		grid_search.fit(X, y)
		Final_model = grid_search.best_estimator_
		Best_parameter = grid_search.best_params_
		Best_score = grid_search.best_score_
		cv_proba = cross_val_predict(estimator=Final_model, X=X, y=y, cv=int(args.cv_num), method='predict_proba')
		cv_proba = pd.DataFrame(cv_proba)
		clf = DefineClf_RandomForest(Best_parameter['n_estimators'],Best_parameter['max_depth'],Best_parameter['max_features'])
		Imp_per = IMP_permutation(clf, cv_num=5, X=X, y=y, score = args.score)

	# draw auc roc for CV
	cv_proba = pd.DataFrame(cv_proba)
	if args.smote == 'y':
		fpr, tpr, thresholds = roc_curve(y_upsample_CV, cv_proba[1])
		# AUC PR
		average_precision = average_precision_score(y_upsample_CV, cv_proba[1])
		precision, recall, _ = precision_recall_curve(y_upsample_CV,cv_proba[1])
	else:
		fpr, tpr, thresholds = roc_curve(y, cv_proba[1])
		# AUC PR
		average_precision = average_precision_score(y, cv_proba[1])
		precision, recall, _ = precision_recall_curve(y,cv_proba[1])
	plt.figure()
	plt.step(recall, precision, where='post')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	if args.feat != 'all':
		plt.savefig(args.file + '_RF_cv%s_score_%s_%s_AUCPR_AP_%s_%s.pdf'%(args.cv_num,args.score,args.smote,average_precision,args.feat))
	else:
		plt.savefig(args.file + '_RF_cv%s_score_%s_%s_AUCPR_AP_%s.pdf'%(args.cv_num,args.score,args.smote,average_precision))

	# AUC_ROC
	plt.figure(figsize=(5,5))
	lw = 2
	plt.plot(fpr, tpr, color='darkorange',lw=lw, label='ROC curve (area = %0.2f)' % auc(fpr,tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('Receiver operating characteristic example')
	plt.legend(loc="lower right")
	if args.feat != 'all':
		plt.savefig(args.file + '_RF_cv%s_score_%s_%s_%s_AUCROC.pdf'%(args.cv_num,args.score,args.smote,args.feat))
	else:
		plt.savefig(args.file + '_RF_cv%s_score_%s_%s_AUCROC.pdf'%(args.cv_num,args.score,args.smote))
		
	if args.smote == 'y':
		#threshold = get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds,y_upsample.value_counts()[1],y_upsample.value_counts()[0])
		threshold = get_metric_and_best_threshold_from_roc_curve_best_measure(y_upsample_CV,cv_proba,'f1')
	else:
		#threshold = get_metric_and_best_threshold_from_roc_curve(tpr, fpr, thresholds,y.value_counts()[1],y.value_counts()[0])
		threshold = get_metric_and_best_threshold_from_roc_curve_best_measure(y,cv_proba,'f1')

	#cv_pred = cross_val_predict(estimator=grid_search.best_estimator_, X=X, y=y, cv=cv_num)
	cv_pred = np.where(cv_proba[1] > threshold, 1, 0)

	# pre_test = grid_search.best_estimator_.predict(X_test)
	# prob_test = grid_search.best_estimator_.predict_proba(X_test)
	if args.feat != 'all':
		joblib.dump(Final_model, args.file + '_RF_cv%s_score_%s_%s_%s.pkl'%(args.cv_num,args.score,args.smote,args.feat))
	else:
		joblib.dump(Final_model, args.file + '_RF_cv%s_score_%s_%s.pkl'%(args.cv_num,args.score,args.smote))

	# output the feature importance from RF entropy based importance
	imp = pd.DataFrame({'Feature':X.columns, 'Importance':Final_model.feature_importances_})
	imp_sorted = imp.sort_values(by='Importance', ascending=False)
	imp_sorted_dir = Imp_dir(X, y, imp_sorted)
	if args.feat != 'all':
		imp_sorted_dir.to_csv(args.file + '_RF_cv%s_score_%s_%s_%s_imp'%(args.cv_num,args.score, args.smote,args.feat), index=False, header=True,sep="\t")
	else:
		imp_sorted_dir.to_csv(args.file + '_RF_cv%s_score_%s_%s_imp'%(args.cv_num,args.score, args.smote), index=False, header=True,sep="\t")
	
	# output the permutation importance
	Mean = Imp_per.mean(axis=1)
	res = pd.concat([Imp_per,pd.DataFrame(Mean)],axis=1)
	res.columns = ['CV_1','CV_2','CV_3','CV_4','CV_5','Mean']
	res = res.sort_values(by=['Mean'],ascending=False)
	if args.feat != 'all':
		res.to_csv(args.file + '_RF_cv%s_score_%s_%s_%s_permutation_imp'%(args.cv_num,args.score, args.smote,args.feat), index=True, header=True,sep="\t")
	else:
		res.to_csv(args.file + '_RF_cv%s_score_%s_%s_permutation_imp'%(args.cv_num,args.score, args.smote), index=True, header=True,sep="\t")

	if args.feat != 'all':
		out = open(args.file + '_RF_cv%s_score_%s_%s_%s'%(args.cv_num,args.score, args.smote,args.feat),'w')
	else:
		out = open(args.file + '_RF_cv%s_score_%s_%s'%(args.cv_num,args.score, args.smote),'w')
	out.write('Best parameters:\n')
	if args.smote == 'y':
		out.write('\tmax_depth:%s\n'%Best_parameter['max_depth'])
		out.write('\tmax_features:%s\n'%Best_parameter['max_features'])
		out.write('\tn_estimators:%s\n\n'%Best_parameter['n_estimators'])
		out.write('Best CV score: %s\n\n'%Best_score)
		out.write('Best threshold: %s\n\n'%threshold)
		out.write('AucRoc_CV_GS: %s\n\n'%AUCROC[parameter])
		#out.write('AucRoc_CV: %s\n\n'%roc_auc_score(y_upsample,cv_pred))
		out.write('AucRoc_CV: %s\n\n'%roc_auc_score(y_upsample_CV,cv_pred))
		out.write('F1_CV_GS: %s\n\n'%F1_score[parameter])
		#out.write('F1_CV: %s\n\n'%f1_score(y_upsample,cv_pred))
		out.write('F1_CV: %s\n\n'%f1_score(y_upsample_CV,cv_pred))
		out.write('Accuracy_CV_GS: %s\n\n'%Accuracy_score[parameter])
		#out.write('Accuracy_CV: %s\n\n'%accuracy_score(y_upsample,cv_pred))
		out.write('Accuracy_CV: %s\n\n'%accuracy_score(y_upsample_CV,cv_pred))
	else:
		out.write('\tmax_depth:%s\n'%Best_parameter['max_depth'])
		out.write('\tmax_features:%s\n'%Best_parameter['max_features'])
		out.write('\tn_estimators:%s\n\n'%Best_parameter['n_estimators'])
		out.write('Best CV score: %s\n\n'%Best_score)
		out.write('Best threshold: %s\n\n'%threshold)
		out.write('AucRoc_CV: %s\n\n'%roc_auc_score(y,cv_pred))
		out.write('F1_CV: %s\n\n'%f1_score(y,cv_pred))
		out.write('Accuracy_CV: %s\n\n'%accuracy_score(y,cv_pred))
	out.close()

if __name__ == '__main__':
	main()




