# DFELMDA: Identification of miRNA–disease associations via deep forest ensemble learning based on autoencoder

Author：Wei Liu, Hui Lin, Li Huang, Li Peng, Ting Tang, Qi Zhao, Li Yang
doi：https://doi.org/10.1093/bib/bbac104

## Abstract
Increasing evidences show that the occurrence of human complex diseases is closely related to microRNA (miRNA) variation and imbalance. For this reason, 
predicting disease-related miRNAs is essential for the diagnosis and treatment of complex human diseases. Although some current computational methods can effectively 
predict potential disease-related miRNAs, the accuracy of prediction should be further improved. In our study, a new computational method via deep forest ensemble 
learning based on autoencoder (DFELMDA) is proposed to predict miRNA–disease associations. Specifically, a new feature representation strategy is proposed to obtain 
different types of feature representations (from miRNA and disease) for each miRNA–disease association. Then, two types of low-dimensional feature representations 
are extracted by two deep autoencoders for predicting miRNA–disease associations. Finally, two prediction scores of the miRNA–disease associations are obtained by 
the deep random forest and combined to determine the final results. DFELMDA is compared with several classical methods on the The Human microRNA Disease Database 
(HMDD) dataset. Results reveal that the performance of this method is superior. The area under receiver operating characteristic curve (AUC) values obtained by 
DFELMDA through 5-fold and 10-fold cross-validation are 0.9552 and 0.9560, respectively. In addition, case studies on colon, breast and lung tumors of different 
disease types further demonstrate the excellent ability of DFELMDA to predict disease-associated miRNA–disease. Performance analysis shows that DFELMDA can be used 
as an effective computational tool for predicting miRNA–disease associations.
