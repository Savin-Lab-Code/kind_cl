# kind_cl


A repo for kindergarten curriculum learning in RNNs. 

![alt text](kind.png)


## Basic repo structure

- dynamics: main python module
  - process/rnn/: core code for training and simulating data from RNNs
  - analysis: behavioral, state-space, and dynamics analsyis of RNNs
  - utils: default configurations, RNN input specs, helper files
  - vis: helper plotting functions
- demo: RNN training demos. for training and stimulating from an RNN model
- vis: Figure generation scripts


## Basic data structure
-rat_data: data from Mah et al. Zenodo repository, containing rat behavioral data
-rnn_data: trained RNN model files and behavioral data, organized by CL type
    - full_cl: kindergarten CL + shaping
    - nok_cl: shaping only
    - nok_nocl: no shaping
    
    
### RNN data file types
It is impossible to include full training snapshots for all of the networks, given the prohibitively large file sizes. However we have included intermediate-level, behavior-only data, as well as a few examples of RNN activity data from select neurons. The file types that are included are the following
- .model: a trained RNN model file that can be loaded with pytorch
- .stats: a dictionary containing summary-level wait-time data for a model
- _1k.json: example of neural activity and behavioral data for an RNN. Included only for RNNs 33,16,9. Output of simulating from RNN
- _allbeh.json: aggregated behavioral data across trainign for an RNN. does NOT include RNN activity

