# pac-bayes tree

Reference: https://papers.nips.cc/paper/8158-pac-bayes-tree-weighted-subtrees-with-guarantees.pdf

Classes:
- helpers:
    + example: Node_A.py 
- PAC-Bayes tree: best settings are adaptive penalty, effective intersection prior and error-first search of temperature parameter 
    + example: dyadic_pacbayes.py 
- pruning: 
    + example: PDDT_master.py
- Helmbold-Schapire aggregate 
    + example: HSDDT_master.py

Results: each .npy file stores either the classification accuracy on the test set or the margin distribution on the test set
of each experiment (an experiment is a split of the sample into training/testing). 

Reports: a jupyter notebook demonstrating how the .npy files in Results can be visualized as plots. 

Datasets: 
- UCI datasets: 
    + spam: spambase.data 
    + digit: optdigits.tra and optdigits.tes 

Scripts:
- fit PAC-Bayes tree, evaluate classification performance and report margin distribution:
    + example: dataMaster_[dyadic_pacbayes]_[eTS,MD].py 

Future steps:
- replicate results in paper
    + get dydic_pacbayes.py to run on spam dataset 
- what's the distinction between Imp_AkdDT_nS_master and AkdDT_nS_master? 
- unify classes for dyadic trees and kd-trees:
    +  after construction of the template tree T0
- unify Node_A.py and Node_P.py 