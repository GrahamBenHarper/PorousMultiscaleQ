# PorousMultiscaleQ
Using deal.II with FE_Q elements to compute a finite element solution to a porous multiscale problem.

Here are some example results for the SinSin example in 2-dimensions as of commit 2ad7563. L2 errors of p are in the first table, and runtimes (in seconds) are in the second table. Clearly the fine refinements are not yet working as intended, so this will be updated again once the results look more reliable.

Fine\Coarse refinements | 1        | 2         | 3         | 4         | 5         
------------------------|----------|-----------|-----------|-----------|-----------
1                       | 0.191558 | 0.0642159 | 0.0261266 | 0.0121842 | 0.00597665
2                       | 0.303505 | 0.108429  | 0.0415316 | 0.0186115 | 0.00900845
3                       | 0.355483 | 0.135340  | 0.0504894 | 0.0220237 | 0.0105509 
4                       | 0.378436 | 0.149672  | 0.0553026 | 0.0237887 | 0.0113302 
5                       | 0.389045 | 0.157010  | 0.0577936 | 0.0246870 | 0.0117221 


Fine\Coarse refinements | 1     | 2     | 3     | 4     | 5
------------------------|-------|-------|-------|-------|-------
1                       | 0.004 | 0.006 | 0.024 | 0.102 | 0.447
2                       | 0.003 | 0.009 | 0.035 | 0.142 | 0.618
3                       | 0.007 | 0.020 | 0.070 | 0.277 | 1.169
4                       | 0.019 | 0.055 | 0.213 | 0.771 | 3.111
5                       | 0.077 | 0.227 | 0.807 | 3.100 | 12.591

