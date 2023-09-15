import pandas as pd
import numpy as np
import sys,os,argparse
from sklearn import datasets
from sklearn.metrics import f1_score 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import roc_auc_score 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve, auc
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import average_precision_score

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def main():
	parser = argparse.ArgumentParser(description='This code is used to apply a model to test set. ')
	# Required
	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-model', help='model needs to be applied to test set', required=True)
	req_group.add_argument('-new_file', help='test set', required=True)
	req_group.add_argument('-threshold', help='threshold to gain the best score, reported by the model', required=True)
	req_group.add_argument('-save_file', help='file name to save', required=True)
	# optional
	inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
	inp_group.add_argument('-feat', help='files with selected features',default = 'all')
	inp_group.add_argument('-smote', help='up-sampling the minority class or not, y or n',default = 'n')

	if len(sys.argv)==1:
		parser.print_help()
		sys.exit(0)
	args = parser.parse_args()	

	# model = args.model
	# new_file = args.new_file
	# threshold = float(args.threshold)
	# save_file = args.save_file

	my_model = joblib.load(args.model)
	df = pd.read_csv(args.new_file, sep='\t',header=0,index_col=0)
	if args.feat != 'all':
		print('Using subset of features from: %s' % args.feat)
		with open(args.feat) as f:
			features = f.read().strip().splitlines()
			features = ['Class'] + features
		df = df.loc[:, features]

	X_new = df.drop('Class',axis=1)
	y_new = df.Class
	if args.smote == 'y':
		from imblearn.over_sampling import SMOTE
		smoter = SMOTE(sampling_strategy='not majority',random_state=42,k_neighbors=3)
		X_new_upsample, y_new_upsample = smoter.fit_resample(X_new,y_new)
		X_new = X_new_upsample
		y_new = y_new_upsample

	pred_new = my_model.predict(X_new)
	prob_new = my_model.predict_proba(X_new)
	prob_new = pd.DataFrame(prob_new)
	# draw the auc-PR
	average_precision = average_precision_score(y_new, prob_new[1])
	precision, recall, _ = precision_recall_curve(y_new,prob_new[1])
	plt.figure()
	plt.step(recall, precision, where='post')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.ylim([0.0, 1.05])
	plt.xlim([0.0, 1.0])
	plt.savefig(args.save_file + '_AUCPR_AP_%s.pdf'%(average_precision))


	# draw the auc-ROC
	fpr, tpr, thresholds = roc_curve(y_new, prob_new[1])
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
	plt.savefig(args.save_file + '_AUCROC.pdf')

	pre_new = np.where(prob_new[1] > float(args.threshold), 1, 0)
	accuracy = accuracy_score(y_new,pre_new)
	F1 = f1_score(y_new,pre_new)
	CM_based_on_threshold = confusion_matrix(y_new,pre_new)
	CM = confusion_matrix(y_new,pred_new)

	out = open(args.save_file,'w')
	out.write('F1 new based on threshold: %s\n\n'%F1)
	out.write('F1 new based on prediction: %s\n\n'%f1_score(y_new,pred_new))
	out.write('Accuracy new based on threshold: %s\n\n'%accuracy)
	out.write('Accuracy new based on prediction: %s\n\n'%accuracy_score(y_new,pred_new))
	out.write('AUC ROC new based on threshold: %s\n\n'%roc_auc_score(y_new,pre_new))
	out.write('AUC ROC new based on prediction: %s\n\n'%roc_auc_score(y_new,pred_new))
	out.write('Confusion matrix based on threshold: \n\n')
	out.write('\t\tPositive\tNegative\n')
	out.write('\tPositive\t%s\t%s\n'%(CM_based_on_threshold[1,1],CM_based_on_threshold[1,0]))
	out.write('\tNegative\t%s\t%s\n'%(CM_based_on_threshold[0,1],CM_based_on_threshold[0,0]))
	out.write('Confusion matrix based on prediction: \n\n')
	out.write('\t\tPositive\tNegative\n')
	out.write('\tPositive\t%s\t%s\n'%(CM[1,1],CM[1,0]))
	out.write('\tNegative\t%s\t%s\n'%(CM[0,1],CM[0,0]))
	out.close()

	pred = y_new.copy()
	pred = pd.DataFrame(pred)
	pre_new = pd.DataFrame(pre_new) # prediction based on threshold
	pre_new.index = pred.index
	pred = pd.concat([pred,pre_new],axis=1)
	pred_new = pd.DataFrame(pred_new) # prediction given by the model, i.e., threhold at 0.5
	pred_new.index = pred.index
	pred = pd.concat([pred,pred_new],axis=1)
	prob_new.index = pred.index # probability of a given gene to be predicted as 1 or 0
	pred = pd.concat([pred,prob_new],axis=1)
	pred.columns = ['Class','Prediction_based_on_threshold_%s'%args.threshold,'Prediction_based_on_0.5','Probability_of_being_predicted_as_negative','Probability_of_being_predicted_as_positive']
	pred.to_csv(args.save_file + '_prediction.txt', index=True, header=True,sep="\t")



if __name__ == '__main__':
	main()



