# DSC_baseline
Code to iteratively subtract the baseline from DSC data, based on Ref. [1]. 

# Usage
`python DSC_baseline.py xfile.format yfile.format t_beg t_end iterations`
where
1. `xfile.format` and `yfile.format` are the x and y components of the DSC data;
2. `t_beg` is the index where the initial linear fit stops;
3. `t_end` is the index where the final linear fit begins; 
4. `iterations` is the number of iterations to perform. 

# References
[1] To be added. 
