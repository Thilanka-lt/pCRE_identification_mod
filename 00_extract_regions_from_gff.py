import sys,pprint,os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from random import randint
from Bio.Seq import Seq
import argparse


def get_sequnce_plus(chm_seq,up_start,down_end,gene_start,gene_end,region,region_len):
    '''
    This is a function to get squnce region depending on the user specification of up, down and gene body of a gene
    This function can get 
        1. only upstrem and down strem region
        2. upstrem and downstream region including whole gene body
        3. Upstrem and prefered length into gene, prefered length to TSS with downstrem
    MUST USE THIS FUNCTION ON GENES THAT ARE ON THE PLUS STRND
    '''
    if region == 1:
        trans_seq = (chm_seq[up_start:(gene_start-1)] + chm_seq[gene_end:down_end]).seq
    elif region == 2:
        trans_seq = seqs[up_start:down_end].seq
    elif region == 3:
        if (gene_end - gene_start) + 1 > region_len * 2: #checking if the length of the gene is less than preferred lenth into the gene
            trans_seq = (chm_seq[up_start:(gene_start+region_len-1)] + chm_seq[(gene_end - region_len):down_end]).seq
        else:
            trans_seq = seqs[up_start:down_end].seq
    return trans_seq
            
        
def get_sequnce_minus(chm_seq,up_start,down_end,gene_start,gene_end,region,region_len):
    '''
    This is a function to get squnce region depending on the user specification of up, down and gene body of a gene
    This function can get 
        1. only upstrem and down strem region
        2. upstrem and downstream region including whole gene body
        3. Upstrem and prefered length into gene, prefered length to TSS with downstrem
    MUST USE THIS FUNCTION ON GENES THAT ARE ON THE MINUS STRND
    '''
    if region == 1:
        trans_seq = (chm_seq[down_end:gene_start-1] + chm_seq[gene_end:up_start]).seq
    elif region == 2:
        trans_seq = chm_seq[down_end:up_start].seq
    elif region ==3:
        if (gene_end - gene_start) + 1 > region_len * 2:
            trans_seq = (chm_seq[down_end:(gene_start+region_len-1)] + chm_seq[gene_end - region_len:up_start]).seq
        else:
            trans_seq = chm_seq[down_end:up_start].seq
    return trans_seq
            
parser = argparse.ArgumentParser(description='This a code to extract up stream downstrem and given length into the sequnce using GFF and genome sequnce file')
parser.add_argument('-g', '--gff',required=True, help="name of the gff file. Should be in gff 3 format")
parser.add_argument('-c', '--chr', required=True, help="name of the chromosome file")
parser.add_argument('-u', '--up', required=True,type=int, help="upstream region that needs to be extrated")
parser.add_argument('-d', '--down', required=True, type=int, help="downstream region that needs to be extrated")
parser.add_argument('-r', '--region', required=True, type=int, help="1 if considering only up and downstream region, 2 if considring up + full gene + downstream region, 3 if considring up + TSS + region + region + TTS+ downstrem region")
parser.add_argument('-i', '--inr', required=False, default=0, type=int, help="region of the gene that needs to be included")
parser.add_argument('-save', '--save_path', required=False, default="./", help="path to save the output file")
args = parser.parse_args()

gff_file_nm = args.gff
chromosome_file = args.chr
up5 =  args.up 
down3 = args.down
gene_region = args.region
in_gene = args.inr
#save_path = args.save_path


gff_file = open(gff_file_nm,"r").readlines()
Gene_info = {}
D = {}
for lines in gff_file:
    if not lines.startswith("#"):
        if lines.split("\t")[2].lstrip().rstrip() == "gene":
            info = lines.split("\t")
            #print(info) 
            gene_id = info[8].split(";")[0].replace("ID=","").lstrip().rstrip()
            strat = int(info[3].lstrip().rstrip())
            end = int(info[4].lstrip().rstrip())
            direction = info[6].lstrip().rstrip()
            chromosome = info[0].lstrip().rstrip()
            #print(f'{gene_id}\t{chromosome}\t{strat}\t{end}\t{direction}')
            if chromosome not in D:
                D[chromosome]=[]
            D[chromosome].append([gene_id,strat,end,direction]) #gene name start_position end_position direction_of_transcription



#position = open("ITAG4.0_gene_models.gff_longed_mRNA.txt_start_end.txt","r").readlines()
#D = {}
#for line in position:
#        tem = line.strip().split("\t")
#        chrom = tem[0]
#        if chrom not in D:
#                D[chrom]=[]
#        D[chrom].append([tem[1],tem[2],tem[3],tem[4]]) #gene name start_position end_position direction_of_transcription

Genome= SeqIO.parse(open(chromosome_file),'fasta')

#out = open("test_Tomato_full_genic_seq_for_pCRE_%s_%s.txt"%(up5, down3),"w")
if args.region == 1:
    save_name = os.path.join(args.save_path ,f'{chromosome_file.split("/")[-1]}_upstream_{up5}_ingene_0_downstream_{down3}.fas')
elif args.region == 2:
    save_name = os.path.join(args.save_path ,f'{chromosome_file.split("/")[-1]}_upstream_{up5}_full_gene_downstream_{down3}.fas')
elif args.region == 3:
    save_name = os.path.join(args.save_path ,f'{chromosome_file.split("/")[-1]}_upstream_{up5}_ingene_{in_gene}_downstream_{down3}.fas')
else:
    print("Please enter 1 or 2 or 3 for -r option")
    sys.exit()

out2 = open(save_name,"w")
for seqs in Genome:
        if seqs.id in D.keys():
                for trans in D[seqs.id]:
                        trans_id = trans[0]
                        S = int(trans[1]) ##trans start point
                        E = int(trans[2]) ##trans end point
                        direction = trans[3]
                        if direction == '+':
                                if (int(S) > up5) & ((int(E) + down3) < len(seqs.seq)): #if gene's upstrem and downstrem region does not exceed the chromosome length  
                                        trans_up_start = S - up5 - 1
                                        trans_down_end = E + down3
                                        #print(trans_up_start,trans_down_end)
                                        trans_seq = get_sequnce_plus(seqs,trans_up_start,trans_down_end,S,E,gene_region,in_gene)
                                        
                                        
                                elif int(S) < up5: #if upstrem is larger than firt bp of chromosome
                                        trans_up_start = 0
                                        trans_down_end = E + down3
                                        trans_seq = get_sequnce_plus(seqs,trans_up_start,trans_down_end,S,E,gene_region,in_gene)
                                elif (int(E) + down3) > len(seqs.seq): #if downstrem is larger than end of the chromosome
                                        trans_up_start = S - up5 - 1
                                        trans_down_end = len(seqs.seq)
                                        trans_seq = get_sequnce_plus(seqs,trans_up_start,trans_down_end,S,E,gene_region,in_gene)
                                else:
                                        print(f"check sequnce {trans_id.split(':')[1]} direction:{direction} start:{S} end:{E} chromosome_length: {len(seqs.seq)}")
                        if direction == '-':
                                if ((int(E) + up5) < len(seqs.seq)) & (int(S) > down3): ##if gene's upstrem and downstrem region does not exceed the chromosome length
                                        trans_up_start = E + up5
                                        trans_down_end = S - down3 -1
                                        a = get_sequnce_minus(seqs,trans_up_start,trans_down_end,S,E,gene_region,in_gene)
                                elif (int(E) + up5) > len(seqs.seq): ##if upstrem is larger than last bp of chromosome
                                        trans_up_start = len(seqs.seq)
                                        trans_down_end = S - down3 -1
                                        a = get_sequnce_minus(seqs,trans_up_start,trans_down_end,S,E,gene_region,in_gene)
                                elif int(S) <  down3: #if downstrem is larger than start of the chromosome
                                        trans_up_start = E + up5
                                        trans_down_end = 0
                                        a = get_sequnce_minus(seqs,trans_up_start,trans_down_end,S,E,gene_region,in_gene)
                                else:
                                        print(f"check sequnce {trans_id.split(':')[1]} direction:{direction} start:{E} end:{S} chromosome_length: {len(seqs.seq)}")
                                trans_seq = a.reverse_complement()
                        #out.write("%s\t%s\n"%(trans_id,trans_seq))
                        out2.write(">%s\n%s\n"%(trans_id.split(':')[1],trans_seq))
out2.close()
