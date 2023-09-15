import sys,os,argparse
import pandas as pd
import numpy as np
import random
from statsmodels.stats.multitest import multipletests as multi
from sklearn.model_selection import StratifiedKFold

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def main():
	########################
	### Parse Input Args ###
	########################
	parser = argparse.ArgumentParser(description='This code contains the kmer enrichment analysis and the following RF model building. Example for using this: python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/000_pipeline_for_holdout_MotifDiscovery_add_SMOTE.py -pos_ID Swithchgrass_Chilling_up_only_D2_gene_list_1887_short.txt -neg_ID Swithchgrass_nochange_FC0.5_FDR0.1_gene_list_short.txt -fasta Switchgrass_seq_for_pCRE_up5_1000_down5_500_up3_500_down3_1000.fas -bal n -holdout 50 -kmers 5mers_withRC.txt -pval 0.01 -smote y -cv_num 5 -score roc_auc')

	### Input arguments ###
	# Required
	req_group = parser.add_argument_group(title='REQUIRED INPUT')
	req_group.add_argument('-pos_ID', help='file with gene ID used for positive class', required=True)
	req_group.add_argument('-neg_ID', help='file with gene ID used for negative class', required=True)
	req_group.add_argument('-fasta', help='DNA sequences used for all the genes', required=True)
	req_group.add_argument('-bal', help='balance the number of positive and negative classes or not, e.g., y or n', required=True)
	req_group.add_argument('-holdout', help='proportion you want to holdout as test, e.g., 0.1 or 0.2, or numbers of instance you want to holdout for positive and negative, e.g., 30 or 50', required=True)
	req_group.add_argument('-kmers', help='list of kmers,delimited by comma, e.g., 5mers_withRC.txt,6mers_withRC.txt,7mers_withRC.txt,8mers_withRC.txt,5mer_pairs_withRC.txt', required=True)
	req_group.add_argument('-pval', help='pvalue threshold for fisher exact test', required=True)
	
	# optional
	inp_group = parser.add_argument_group(title='OPTIONAL INPUT')
	inp_group.add_argument('-padjust', help='do the padjust or not, y or n',default='n')
	inp_group.add_argument('-qval', help='qvalue threshold for fisher exact test',default=0.05)
	inp_group.add_argument('-smote', help='up-sampling the minority class or not, y or n',default='n')
	inp_group.add_argument('-cv_num', help='number of folds for cross validation',default=5)
	inp_group.add_argument('-score', help='score used to select final model',default='roc_auc')
	inp_group.add_argument('-smote_test', help='',default='up-sampling the minority class or not for test dataset, y or n')
	inp_group.add_argument('-stratified_for_test', help='y or n, y only when  the holdout < 1, i.e., proportion to be treated as test', default='n')
	inp_group.add_argument('-pos_expression', help='file with gene ID and used for positive class and the expression, can be TPM or FC or any values, needed when stratified_for_test is y', default='none')
	inp_group.add_argument('-neg_expression', help='file with gene ID and used for negative class and the expression, can be TPM or FC or any values, needed when stratified_for_test is y', default='none')
	inp_group.add_argument('-column', help='column used for the expression file, needed when stratified_for_test is y', default='none')
	inp_group.add_argument('-short', help='this is for switchgrass, specifically, where the gene names would be shorten. y or n', default='n')
	inp_group.add_argument('-alg', help='ML algorithms', default='RF')
	inp_group.add_argument('-gs_reps', help='how many replicates for grid search', default=10)
	inp_group.add_argument('-n_reps', help='how many replicates for down sampling', default=10)
	inp_group.add_argument('-n_jobs', help='Number of processors for parallel computing (max for HPCC = 14)', default=4)
	
	
	if len(sys.argv)==1:
		parser.print_help()
		sys.exit(0)
	args = parser.parse_args()	
	
	pos_ID = args.pos_ID
	neg_ID = args.neg_ID
	Pos_ID = pd.read_csv('FastaFiles/' + pos_ID,header=None,sep='\t')
	Neg_ID = pd.read_csv('FastaFiles/' + neg_ID,header=None,sep='\t')

	# balance the number of positive and negative classes
	if args.bal == 'y':
		if Pos_ID.shape[0] < Neg_ID.shape[0]:
			Neg_ID = Neg_ID.sample(n=Pos_ID.shape[0],replace=False,random_state=42)
			Neg_ID.to_csv('FastaFiles/' + neg_ID.split('.txt')[0] + '_%s.txt'%Pos_ID.shape[0], header=False,index=False,sep='\t')
			neg_ID = neg_ID.split('.txt')[0] + '_%s.txt'%Pos_ID.shape[0]
		else:
			Pos_ID = Pos_ID.sample(n=Neg_ID.shape[0],replace=False,random_state=42)
			Pos_ID.to_csv('FastaFiles/' + pos_ID.split('.txt')[0] + '_%s.txt'%Neg_ID.shape[0], header=False,index=False,sep='\t')
			pos_ID = pos_ID.split('.txt')[0] + '_%s.txt'%Neg_ID.shape[0]

	# hold out certain number of instance as test
	# randomly select the test genes
	if args.stratified_for_test == 'n':
		if float(args.holdout) > 1: # numbers
			pos_test_ID = Pos_ID.sample(n=int(args.holdout),replace=False,random_state=42)
			neg_test_ID = Neg_ID.sample(n=int(args.holdout),replace=False,random_state=42)
			number_holdout_pos = int(args.holdout)
			number_holdout_neg = int(args.holdout)
			number_training_pos = Pos_ID.shape[0] - int(args.holdout)
			number_training_neg = Neg_ID.shape[0] - int(args.holdout)
		if float(args.holdout) < 1: # percentage
			number_holdout_pos = int(Pos_ID.shape[0] * float(args.holdout))
			number_holdout_neg = int(Neg_ID.shape[0] * float(args.holdout))
			number_training_pos = Pos_ID.shape[0] - int(Pos_ID.shape[0] * float(args.holdout))
			number_training_neg = Neg_ID.shape[0] - int(Neg_ID.shape[0] * float(args.holdout))
		pos_test_ID = Pos_ID.sample(n=number_holdout_pos,replace=False,random_state=42)
		neg_test_ID = Neg_ID.sample(n=number_holdout_neg,replace=False,random_state=42)
		pos_train_ID = pd.concat([Pos_ID, pos_test_ID, pos_test_ID]).drop_duplicates(keep=False)
		neg_train_ID = pd.concat([Neg_ID, neg_test_ID, neg_test_ID]).drop_duplicates(keep=False)
	else:
		# do the stratified selection for the test genes
		pos_expression = args.pos_expression
		neg_expression = args.neg_expression
		column = args.column
		Pos_exp = pd.read_csv('FastaFiles/' + pos_expression,header=0,sep='\t',index_col=0)
		Neg_exp = pd.read_csv('FastaFiles/' + neg_expression,header=0,sep='\t',index_col=0)
		if args.short == 'y':
			Pos_exp.index = Pos_exp.index.str.replace('Pavir.','')
			Pos_exp.index = Pos_exp.index.str.replace('.v5.1','')
			Neg_exp.index = Neg_exp.index.str.replace('Pavir.','')
			Neg_exp.index = Neg_exp.index.str.replace('.v5.1','')
		Pos_exp['Class'] = 1
		Pos_exp = Pos_exp.loc[:,['Class',column]]
		Neg_exp['Class'] = 0
		Neg_exp = Neg_exp.loc[:,['Class',column]]
		Df = pd.concat([Pos_exp,Neg_exp],axis=0)
		Tr_te = StratifiedKFold(n_splits=int(1/float(args.holdout)), random_state=42,shuffle=True)
		nn = 0
		for train_index, test_index in Tr_te.split(Df.drop('Class',axis=1),Df.Class):
			if nn == 0:
				train_ID = Df.iloc[train_index]
				test_ID = Df.iloc[test_index]
				nn = 1
		pos_train_ID = pd.DataFrame(train_ID[train_ID.Class==1].index.tolist())
		neg_train_ID = pd.DataFrame(train_ID[train_ID.Class==0].index.tolist())
		pos_test_ID = pd.DataFrame(test_ID[test_ID.Class==1].index.tolist())
		neg_test_ID = pd.DataFrame(test_ID[test_ID.Class==0].index.tolist())
		number_holdout_pos = len(pos_test_ID)
		number_holdout_neg = len(neg_test_ID)
		number_training_pos = len(pos_train_ID)
		number_training_neg = len(neg_train_ID)
	pos_test_ID.to_csv('FastaFiles/test_pos_geneID_%s.txt'%number_holdout_pos, header=False,index=False,sep='\t')
	pos_train_ID.to_csv('FastaFiles/train_pos_geneID_%s.txt'%number_training_pos, header=False,index=False,sep='\t')
	neg_test_ID.to_csv('FastaFiles/test_neg_geneID_%s.txt'%number_holdout_neg, header=False,index=False,sep='\t')
	neg_train_ID.to_csv('FastaFiles/train_neg_geneID_%s.txt'%number_training_neg, header=False,index=False,sep='\t')
	# get sequences for training and test instances
	print('get sequences for training and test instances')
	pos_test_fa = open('FastaFiles/test_pos_geneID_%s.txt.fa'%number_holdout_pos, 'w')
	pos_train_fa = open('FastaFiles/train_pos_geneID_%s.txt.fa'%number_training_pos, 'w')
	neg_test_fa = open('FastaFiles/test_neg_geneID_%s.txt.fa'%number_holdout_neg, 'w')
	neg_train_fa = open('FastaFiles/train_neg_geneID_%s.txt.fa'%number_training_neg, 'w')

	Seq = open('FastaFiles/' + args.fasta,'r').readlines()
	i = 0
	while i < len(Seq)/2:
		if Seq[i*2].strip()[1:] in pos_test_ID[0].tolist():
			pos_test_fa.write(Seq[i*2])
			pos_test_fa.write(Seq[i*2+1])
		if Seq[i*2].strip()[1:] in pos_train_ID[0].tolist():
			pos_train_fa.write(Seq[i*2])
			pos_train_fa.write(Seq[i*2+1])
		if Seq[i*2].strip()[1:] in neg_test_ID[0].tolist():
			neg_test_fa.write(Seq[i*2])
			neg_test_fa.write(Seq[i*2+1])
		if Seq[i*2].strip()[1:] in neg_train_ID[0].tolist():
			neg_train_fa.write(Seq[i*2])
			neg_train_fa.write(Seq[i*2+1])
		i += 1
	pos_test_fa.close()
	pos_train_fa.close()
	neg_test_fa.close()
	neg_train_fa.close()
	
	# do the cv split
	pos_train_ID['Class'] = 1
	pos_train_ID = pos_train_ID.set_index(0)
	neg_train_ID['Class'] = 0
	neg_train_ID = neg_train_ID.set_index(0)
	train = pd.concat([pos_train_ID,neg_train_ID],axis=0)
	cv = StratifiedKFold(n_splits=args.cv_num, random_state=42,shuffle=True)
	n = 1
	os.system('mkdir Enrichment')
	for train_fold_index, val_fold_index in cv.split(train,train):
		train_fold = train.iloc[train_fold_index]
		# save fasta file for each training fole
		pos_train_fa = open('FastaFiles/train_pos_geneID_%s_cv_%s.txt.fa'%(number_training_pos,n), 'w')
		neg_train_fa = open('FastaFiles/train_neg_geneID_%s_cv_%s.txt.fa'%(number_training_neg,n), 'w')
		Seq = open('FastaFiles/' + args.fasta,'r').readlines()
		i = 0
		while i < len(Seq)/2:
			if Seq[i*2].strip()[1:] in train_fold[train_fold['Class']==1].index.tolist():
				pos_train_fa.write(Seq[i*2])
				pos_train_fa.write(Seq[i*2+1])
			if Seq[i*2].strip()[1:] in train_fold[train_fold['Class']==0].index.tolist():
				neg_train_fa.write(Seq[i*2])
				neg_train_fa.write(Seq[i*2+1])
			i += 1
		pos_train_fa.close()
		neg_train_fa.close()
		# do the enrichment for each training fold
		print('begin to make kmer matrices and do the fisher exact test for %s training fold'%n)
		for kmer in args.kmers.split(','):
			print('working on the %s ... ...'%kmer)
			os.system('python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/MotifDiscovery/Pairwise_kmers.py -f make_df -k /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/MotifLists/%s -p FastaFiles/train_pos_geneID_%s_cv_%s.txt.fa -n FastaFiles/train_neg_geneID_%s_cv_%s.txt.fa'%(kmer, number_training_pos,n, number_training_neg,n))
			k = int(kmer.split('mer')[0])
			if 'pair' not in kmer:
				os.system('python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/MotifDiscovery/Pairwise_kmers.py -f parse_df -df train_pos_geneID_%s_cv_%s_k%s_%s.0single_rc_df.txt -pval %s'%(number_training_pos,n, int(4**k/2),k, args.pval))
			else:
				os.system('python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/MotifDiscovery/Pairwise_kmers.py -f parse_df -df train_pos_geneID_%s_cv_%s_k%s_%s.0paired_rc_df.txt -pval %s'%(number_training_pos, n,int((4**k/2) * (4**k/2 -1)/2),k,args.pval))
			print('Done the %s!'%kmer)
		# merge all the enriched kmers
		# without padjust
		if args.padjust == 'n':
			os.system('cat *_rc_sig_*.txt > Enriched_kmers_cv_%s.txt'%(n))
		# with padjust
		else:
			for file in os.listdir('./'):
				if file.endswith('FETresults.txt'):
					df = pd.read_csv(file,header=0,sep='\t')
					qvalue = multi(df['pvalue'], alpha=0.05, method='fdr_bh')
					df['qvalue'] = qvalue[1]
					enriched = df[df['qvalue'] < float(args.qval)]
					enriched['Kmer'].to_csv(file + '_qval_%s.txt'%args.qval,header=False,index=False,sep='\t')
			os.system('cat *_qval_*.txt > Enriched_kmers_cv_%s.txt'%(n))
			
		# put all the temporary file into a new folder Enrichment
		os.system('mv *_rc_df.txt *_rc_FETresults* *_rc_sig_* Enrichment')
		n += 1
	
	# get the common enriched kmers from all the training folds
	Common = []
	for m in range(1,int(args.cv_num) + 1):
		with open('Enriched_kmers_cv_%s.txt'%m) as f:
			kmers = f.read().strip().splitlines()
		if m == 1:
			Common = kmers
		else:
			Common = list(set(Common) & set(kmers))
	Common_kmers = open('Enriched_kmers.txt','w')
	for k in Common:
		Common_kmers.write(k + '\n')
	Common_kmers.close()

	# remake the matrices for enriched kmers
	print('Begin to make matrices for ML models')
	# train
	os.system('python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/MotifDiscovery/Pairwise_kmers.py -f make_df -k Enriched_kmers.txt -p FastaFiles/train_pos_geneID_%s.txt.fa -n FastaFiles/train_neg_geneID_%s.txt.fa'%(number_training_pos, number_training_neg))
	os.system('mv train_pos* train_enriched_kmers_matrix.txt')
	# test
	os.system('python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/MotifDiscovery/Pairwise_kmers.py -f make_df -k Enriched_kmers.txt -p FastaFiles/test_pos_geneID_%s.txt.fa -n FastaFiles/test_neg_geneID_%s.txt.fa'%(number_holdout_pos, number_holdout_neg))
	os.system('mv test_pos* test_enriched_kmers_matrix.txt')

	# merge training and test set
	os.system("sed -i -e '$a\' test_enriched_kmers_matrix.txt")
	os.system("sed -i -e '$a\' train_enriched_kmers_matrix.txt")
	os.system('head -1 train_enriched_kmers_matrix.txt > whole_matrix.txt; tail -n +2 -q *_enriched_kmers_matrix.txt >> whole_matrix.txt')
	
	# get the test ID
	os.system('tail -n +2 test_enriched_kmers_matrix.txt > test_ids.txt_tem')
	os.system('cut -f1 < test_ids.txt_tem > test_ids.txt')
	os.system('rm test_ids.txt_tem')
	
	# run the RF models
	os.system('python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/ML-Pipeline/ML_classification.py -df whole_matrix.txt -test test_ids.txt -cl_train all -alg %s -apply all -n_jobs %s -n %s -cv_num %s -gs t -gs_reps %s -cm t -plots t '%(args.alg, args.n_jobs, args.n_reps, args.cv_num, args.gs_reps))
	
	# # RandomForestClassifer
	# print('Begin to train the model')
	# os.system('python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/13_RF_holdout_before_enrichment_draw_two_AUC_SMOTE_upsampling_val_02.py -file train_enriched_kmers_matrix.txt -cv_num %s -score %s -smote %s'%(args.cv_num, args.score,args.smote))
	
	# # get the best threshold
	# inp = open('train_enriched_kmers_matrix.txt_RF_cv%s_score_%s_%s'%(args.cv_num, args.score,args.smote),'r')
	# for inl in inp:
		# if inl.startswith('Best threshold: '):
			# threshold = inl.strip().split('Best threshold: ')[1]
	
	# # apply the model on test
	# print('Begin to apply the model on test')
	# os.system('python /mnt/home/peipeiw/Documents/Transformer_Switchgrass/02_MotifDiscovery/09_apply_model_to_new_data_draw_two_AUC_SMOTE_on_test.py -model train_enriched_kmers_matrix.txt_RF_cv%s_score_%s_%s.pkl -new_file test_enriched_kmers_matrix.txt -threshold %s -save_file train_enriched_kmers_applied_on_test_threshold -smote %s'%(args.cv_num, args.score,args.smote,threshold,args.smote_test))
	
	print('Done!')

if __name__ == '__main__':
	main()

