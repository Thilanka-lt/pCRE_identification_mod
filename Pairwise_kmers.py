"""Make data frames needed for Random Forest on single and paires of motifs

FUNCTIONS (-f):

make_pairs 	:=   Make a file with all possible kmer pairs. For 6mers, should have 8,386,560 pairs.
                        Need: -k (# for k)

make_pairs2	:=   Make pairs and singleton file taking into consideration reverse complements. 
                        Need: -k (# for k)

make_df    	:=   Make a table with presence or absense of all kmers/kmer pairs for positive and negative genes. 
                     Works as input for RandomForest in R.
                        Need: -k, -p (fasta files), -n (fasta file)
                        Optional: -ds [file with DNA structure information] *Also be sure to put DS5 as one line in your kmer list!
                         

parse_df	:=   Run Fisher's Exact Test on data in dataframe format (use make_df2 for reverse complement consideration).
                     Outputs: _FETresults.txt (all motifs and pvalues)
                            _sig_0.05.txt (list of motifs that were enriched (pvalue >= 0.05/your cut off)
                      Need: -df
                      Optional: -pval (Default = 0.05)
		
parse		:=   Old version, use parse_df instead.
             Parse table based on enrichment using Fishers Exact Test, default p value is 0.05. Uses df with both pos and neg
                     examples as the imput. 
                      Need: -df
                      Optional: -pval (Default = 0.05)

PARAMETERS AVAILABLE:

-f     :=  functions - defined above
-k     :=  Varies by function. 
             -make_pairs: # of kmer you want
             -make_df: txt file with list of all k-mers/pairs you want in the df
-p      :=  fasta files of positive examples (Will be given Class = 1)
-n      :=  fasta files of negative examples (Will be given Class = 0)
-df     :=  Presence or Absense dataframe (like those made with make_df)
-ds     :=  DNA Structural information (output from /mnt/home/azodichr/scripts/...)
-pval   :=  Default = 0.05



"""

from collections import defaultdict
import sys, os
import itertools


class Kmer_pairs:

	def make_pairs(self,kmers):
		"""Make a file with all possible kmer pairs. For 6mers, should have 8,386,560 pairs"""
		from Bio.Seq import Seq
		#Makes list of all possible kmers
		bases = ['A','T','G','C']
		km = [''.join(p) for p in itertools.product(bases, repeat=int(kmers))]

		#Make all pairwise combinations of the pairs - order does not matter and a kmer does not pair with itself
		pairs =[]
		for i in range(0,len(km)-1):
			for j in range(1,len(km)):
				if i+j < len(km):
					pairs.append(km[i]+ " "+km[i+j])

		print("kmers: "+str(len(km)) + ", Pairs: " + str(len(pairs)))

		out = open(kmers+"_pairs.txt",'w')
		for p in pairs:
			out.write("\n"+p)

###################################################################################################################################

	def make_pairs2(self,kmers):
		"""Make a file with all possible kmer pairs accounting for reverse complements. RC separated by '.', pairs by ' '"""
		from Bio.Seq import Seq
		#Makes list of all possible kmers
		bases = ['A','T','G','C']
		km = [''.join(p) for p in itertools.product(bases, repeat=int(kmers))]
		print("Possible kmers: " + str(len(km)))
		
		#Removes reverse complements so only one version present in list
		for i in km:
			s = Seq(i)
			if s.reverse_complement() in km:
				km.remove(s.reverse_complement())
		print("Kmers (reverse complements removed): " + str(len(km)))

		#Make list of all kmers with their reverse complement 
		rc_list = []
		for j in km:
			revcomp = Seq(j).reverse_complement()
			string = str(j) + "." + str(revcomp)
			rc_list.append(string)

		out = open(kmers+"mers_withRC.txt",'w')
		for k in rc_list:
			out.write("%s\n" % k)

		#Make all pairwise combinations of the pairs - order does not matter and a kmer does not pair with itself
		pairs =[]
		for i in range(0,len(rc_list)-1):
			for j in range(1,len(rc_list)):
				if i+j < len(rc_list):
					pairs.append(rc_list[i]+ " "+rc_list[i+j])

		print("Number of pairs generated: kmers: "+ str(len(pairs)))

		out2 = open(kmers+"mer_pairs_withRC.txt",'w')
		for p in pairs:
			out2.write("%s\n" % p)


	def make_pairs3(self,kmers):
		"""Make a file with all possible kmer pairs accounting for reverse complements. RC separated by '.', pairs by ' '"""
		from Bio.Seq import Seq
		#Makes list of all possible kmers
		bases = ['A','T','G','C']
		km = [''.join(p) for p in itertools.product(bases, repeat=int(kmers))]
		print("Possible kmers: " + str(len(km)))
		
		#Removes reverse complements so only one version present in list
		for i in km:
			s = Seq(i)
			if s.reverse_complement() in km:
				km.remove(s.reverse_complement())
		print("Kmers (reverse complements removed): " + str(len(km)))

		#Make list of all kmers with their reverse complement 
		rc_list = []
		for j in km:
			revcomp = Seq(j).reverse_complement()
			string = str(j) + "." + str(revcomp)
			rc_list.append(string)

		out = open(kmers+"mers_withRC.txt",'w')
		for k in rc_list:
			out.write("%s\n" % k)

	
###################################################################################################################################

	def make_df(self, kmers, pos, neg, ds):
		"""Make a table with presence or absense of all kmers/kmer pairs for positive and negative genes. 
		This version works for kmer list that accound for reverse complements
		For input into randomForest. If inlcuding DNA Structure, include "DS5" in your kmer list"""
		from Bio import SeqIO


		#Put all kmers/kmer pairs into list
		km = []
		for l in open(kmers, 'r'):
			km.append(l.strip("\n"))

		#Get name for saving df, based on positive fasta file name. 
		n = pos.strip().split("/")[-1]

		na = n[:-7]+"_k"+str(len(km))

		##Read positive fasta files into dictionary
		genes = {}
		p = open(pos,'r')
		for seq_record in SeqIO.parse(p, 'fasta'):
			header = seq_record.id
			seq = (str(seq_record.seq))
			genes[header]=seq
		print("Positive Fasta file loaded")

		##Read neg fasta files into dictionary
		genes_neg = {}
		n = open(neg,'r')
		for seq_record in SeqIO.parse(n, 'fasta'):
			header = seq_record.id
			seq = (str(seq_record.seq))
			genes_neg[header]=seq
		print("Negative Fasta file loaded")

		##Read DNA Structure information into dictionary --- if -ds != no
		dsinfo = defaultdict(list)
		if not ds.startswith('no'): 
			for l in open(ds,'r'):
				gene = l.split('\t')[0]
				DNAS = l.strip().split('\t')[1:6]	#Each row has gene name and then the 5 DS principle components
				dsinfo[gene]=DNAS


		##Mark presence or absense of motifs in each sequence in the pos fasta file - for each gene go through all motifs/DNA Structure
		m = 0
		allgenes = defaultdict(list)
		for pi in genes:
			m+=1
			templist = []
			templist.append("1")
			for ki in km:
				if ki=="DS5":			#If you're including DS in dataframe, you should includ "DS5" in your motif list
					info= '\t'.join(dsinfo[pi])
					templist.append(info)	
				elif " " in ki:			#Checks to see if motif is a pair - pairs are separated by a space
					if "." in ki:		#Checks to see if motifs are in reverse complement pairs
						x = (len(ki)-3)/4
						out_name = na+'_'+str(x)+"paired_rc_df.txt"
						kmer1 = ki.split(" ")[0]
						kmer2 = ki.split(" ")[1]
						k1_F = kmer1.split(".")[0]
						k1_R = kmer1.split(".")[1]
						k2_F = kmer2.split(".")[0]
						k2_R = kmer2.split(".")[1]
						seq=genes[pi]
						if k1_F in seq or k1_R in seq and k2_F in seq or k2_R in seq:
							templist.append("1")
						else:
							templist.append("0")
					else:
						x = (len(ki)-1)/2
						out_name = na+'_'+str(x)+"paired_df.txt"
						kmer1 = ki.split(" ")[0]
						kmer2 = ki.split(" ")[1]
						seq = genes[pi]
						if kmer1 in seq and kmer2 in seq:
							templist.append("1")
						else:
							templist.append("0")
				else:					#If not DS5, and no separation by a space, assumes you're looking at singletons.
					if "." in ki:		#Checks to see if motifs are in reverse complement pairs
						x = (len(ki)-1)/2
						out_name = na+'_'+str(x)+"single_rc_df.txt"
						k1_F = ki.split(".")[0]
						k1_R = ki.split(".")[1]
						seq=genes[pi]
						if k1_F in seq or k1_R in seq:
							templist.append("1")
						else:
							templist.append("0")
					else:
						x = len(ki)
						out_name = na+'_'+str(x)+"single_df.txt"
						seq = genes[pi]
						if ki in seq:
							templist.append("1")
						else:
							templist.append("0")
						
			allgenes[pi]=templist
			if m%25==0:
				print("Completed " + str(m) + " positive sequences")
			
		print("All Positive Examples in Dictionary")

		##Mark presence or absense of motifs in each sequence in the neg fasta file - for each gene go through all motifs/DNA Structure

		j = 0
		for ni in genes_neg:
			j+=1
			templist = []
			templist.append("0")
			for ki in km:
				if ki=="DS5":			#If you're including DS in dataframe, you should includ "DS5" in your motif list
					info= '\t'.join(dsinfo[ni])
					templist.append(info)
				elif " " in ki:			#Checks to see if motif is a pair - pairs are separated by a space
					if "." in ki:		#Checks to see if motifs are in reverse complement pairs
						x = (len(ki)-3)/4
						out_name = na+'_'+str(x)+"paired_rc_df.txt"
						kmer1 = ki.split(" ")[0]
						kmer2 = ki.split(" ")[1]
						k1_F = kmer1.split(".")[0]
						k1_R = kmer1.split(".")[1]
						k2_F = kmer2.split(".")[0]
						k2_R = kmer2.split(".")[1]
						seq=genes_neg[ni]
						if k1_F in seq or k1_R in seq and k2_F in seq or k2_R in seq:
							templist.append("1")
						else:
							templist.append("0")
					else: 
						kmer1 = ki.split(" ")[0]
						kmer2 = ki.split(" ")[1]
						seq = genes_neg[ni]
						if kmer1 in seq and kmer2 in seq:
							templist.append("1")
						else:
							templist.append("0")
				else:					#If not DS5, and no separation by a space, assumes you're looking at singletons.
					if "." in ki:		#Checks to see if motifs are in reverse complement pairs
						x = (len(ki)-1)/2
						out_name = na+'_'+str(x)+"single_rc_df.txt"
						k1_F = ki.split(".")[0]
						k1_R = ki.split(".")[1]
						seq=genes_neg[ni]
						if k1_F in seq or k1_R in seq:
							templist.append("1")
						else:
							templist.append("0")
					else:
						seq = genes_neg[ni]
						if ki in seq:
							templist.append("1")
						else:
							templist.append("0")
						
			allgenes[ni]=templist
			if j%25==0:
				print("Completed " + str(j) + " negative sequences")

		print("All Negative Examples in Dictionary")

		out = open(out_name,'w')
		if not ds.startswith('no'):
			out.write("Gene\tClass\t"+"DS1\tDS2\tDS3\tDS4\t"+'\t'.join(km))
		else:
			out.write("Gene\tClass\t"+'\t'.join(km))
		for alli in allgenes:
			out.write("\n"+alli +"\t"+ "\t".join(allgenes[alli]))
		print("Done! # genes:" + str(len(allgenes)))

###################################################################################################################################

	def parse_df(self, df, pval):
		"""Make a table with presence or absense of all kmers/kmer pairs for positive and negative genes. 
		This version works for kmer list that accound for reverse complements
		For input into randomForest. If inlcuding DNA Structure, include "DS5" in your kmer list"""
		from Bio import SeqIO
		from scipy import stats as stats
		from statsmodels.stats.multitest import multipletests as multi
		import pandas as pd
		DF = pd.read_csv(df, sep='\t', index_col = 0, header = 0)
		#Get name for saving df, based on positive fasta file name. 
		n = df.split("/")[-1][:-7]
		p_genes = defaultdict(list)			#These will be dictionary with motif/set as key and list with all genes that have that motif/set as value
		n_genes = defaultdict(list)

		c=0
		for l in open(df,'r'):
			c += 1
			gene = l.strip().split("\t")[0]
			if l.startswith("Gene"):
				motifs = l.strip().split("\t")[2:]
				for k in motifs:
					p_genes[k].append("")
					n_genes[k].append("")
			elif l.strip().split("\t")[1] == "1":		# If Class = 1 (positive examples)
				pos_AbPr = l.strip().split("\t")[2:]
				for i in range(0,len(pos_AbPr)):			# For all motifs/sets of motifs
					if pos_AbPr[i] == "1":							# If the motif is present ('1') put it in p_genes
						p_genes[motifs[i]].append(gene)
			elif l.strip().split('\t')[1] == "0":		# If Class = 0 (negative examples)
				neg_AbPr = l.strip().split("\t")[2:]
				for i in range(0,len(neg_AbPr)):			# For all motifs/sets of motifs
					if neg_AbPr[i] == "1":							# If the motif is present ('1') put it in n_genes
						n_genes[motifs[i]].append(gene)
			else:
				print("Error reading df, couldn't identify Class")

		#num_genes = (c-1)/2
		num_pos = DF[DF['Class']==1].shape[0]
		num_neg = DF[DF['Class']==0].shape[0]
		pos_pres = {}
		neg_pres = {}
		print("Read data into dictionary")

		
		for kmer in p_genes:
			pos_pres[kmer]=len(p_genes[kmer])-1 	#-1 because of the blank "" in the p_genes and n_genes lists. 
			neg_pres[kmer]=len(n_genes[kmer])-1

		outFISH = open(n+"_FETresults.txt",'w')
		outFISH.write('Kmer\tPos_with_Kmer\tPos_with_no_Kmer\tNeg_with_Kmer\tNeg_with_no_Kmer\tpvalue\n')
		outSIG = open(n+"_sig_"+str(pval)+".txt",'w')
		print("Starting Fisher's test")
		count = 0
		for kmer in pos_pres:
			try:
				count += 1 
				oddsratio,pvalue = stats.fisher_exact([[pos_pres[kmer],(num_pos-pos_pres[kmer])],[neg_pres[kmer],(num_neg-neg_pres[kmer])]],alternative='greater')
				outFISH.write(kmer + "\t" +str(pos_pres[kmer])+ "\t" +str(num_pos-pos_pres[kmer])+"\t"+ str(neg_pres[kmer])+"\t"+ str(num_neg-neg_pres[kmer])+ "\t"+ str(pvalue)+"\n")
				if float(pvalue) <= float(pval):
					outSIG.write(kmer+"\n")
				if count%50000==0:
					print("Completed " + str(count) + " kmer pairs.")

			except ValueError:
				count += 1 
				outFISH.write(kmer + "\t" +str(pos_pres[kmer])+ "\t" +str(num_pos-pos_pres[kmer])+"\t"+ str(neg_pres[kmer])+"\t"+ str(num_neg-neg_pres[kmer])+ "\t1\n")

		outFISH.close()
		outSIG.close()


###################################################################################################################################

	
	def parse(self, df, pval):
		"""Parse table based on enrichment using Fishers Exact Test, default p value is 0.05. Uses df with both pos and neg
		examples as the imput"""

		#import pandas as pd
		#import numpy as np
		from scipy import stats as stats

		n = df[:-4]
		out = open(n+"_FET.txt",'w')


		num_pos = 0
		num_neg = 0
		positives = defaultdict(list)
		negatives = defaultdict(list)

		for line in open(df,'r'):
			if line.startswith("Gene"):
				header = line.strip("\n").strip("\r").split("\t")
			else:
				x = line.strip().split("\t")
				if x[1] == "1":
					positives[x[0]]=x[2:]
					num_pos+=1
				if x[1] == "0":
					negatives[x[0]]=x[2:]
					num_neg+=1
		print("Examples loaded into dictionary")

		df_p=pd.DataFrame(positives, dtype="float",index=header[2:])
		df_p = df_p.T 				#Transpose dataframe so motif pairs are separated by column and genes by row.
		print("Positive examples loaded into panda df")
		SUMS_p = df_p.sum(0)
		print("Positive example sums calculated")

		df_n=pd.DataFrame(negatives, dtype="float",index=header[2:])
		df_n = df_n.T 				#Transpose dataframe so motif pairs are separated by column and genes by row.
		print("Negative examples loaded into panda df")
		SUMS_n = df_n.sum(0)
		print("Negative example sums calculated")

		missed = []
		count = 0
		for i in header[2:]:
			count +=1
			try:
				oddsratio,pvalue = stats.fisher_exact([[SUMS_p[i],SUMS_n[i]],[num_pos-SUMS_p[i],num_neg-SUMS_n[i]]])
				out.write(i + "\t" + str(SUMS_p[i])+ "\t" + str(SUMS_n[i])+"\t" + str(pvalue) + "\n")
			except ValueError:
				missed.append(i)
			#if float(pvalue) > float(pval): ###Change back to pvalue instead of 0.5
				#del df_n[i]
				#del df_p[i]

			if count%50000==0:
				print("Completed " + str(count) + " kmer pairs.")
		print (str(len(missed))+": motif pairs were skipped because they had ValueError messages")
		#nname = n+"_neg.csv"
		#pname = n+"_pos.csv"
		#df_p.to_csv(pname)
		#df_n.to_csv(nname)



###################################################################################################################################

if __name__ == "__main__":
	Kmer_pairs=Kmer_pairs()
	pval = '0.05'
	ds = 'no'
	for i in range (1,len(sys.argv),2):
		if sys.argv[i] == "-k":
			kmers = sys.argv[i+1]
		elif sys.argv[i]=="-f":
			F = sys.argv[i+1]
		elif sys.argv[i]=="-df":
			df = sys.argv[i+1]
		elif sys.argv[i]=="-p":
			pos = sys.argv[i+1]
		elif sys.argv[i]=="-n":
			neg = sys.argv[i+1]
		elif sys.argv[i]=="-ds":
			ds = sys.argv[i+1]
		elif sys.argv[i]=="-pval":		#default set at 0.05
			pval = sys.argv[i+1]
		else:
			print("Unknown flag: ",sys.argv[i])
	if len(sys.argv) <= 1:
		print(__doc__)
		sys.exit()

	if F == "make_pairs":
		if "" in [kmers]:
			print("Need files with all k-mers")
		Kmer_pairs.make_pairs(kmers)

	elif F == "make_pairs2":
		if "" in [kmers]:
			print("Need files with all k-mers")
		Kmer_pairs.make_pairs2(kmers)

	elif F == "make_df":
		if "" in [kmers, pos, neg]:
			print("Need kmer list (single or pairs), pos and neg fasta files, and DNA strucuture file if desired.")
		Kmer_pairs.make_df(kmers, pos, neg, ds)    

	elif F == "parse_df":
		if "" in [df]:
			print("Need dataframe.")
		Kmer_pairs.parse_df(df, pval)

	
	elif F == "parse":
		if "" in [df]:
			print("Need presenese/absense data frame with both positive and negative examples (& Class Column)")
		Kmer_pairs.parse(df, pval)
