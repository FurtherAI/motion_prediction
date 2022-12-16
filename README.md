# Pcurvenet
## Install/Environment
Using conda - install pytorch with cuda, pytorch lightning and then use pip to install av2 (Argoverse 2, the dataset's api)
Alternatively, use the environment.yml file to setup the environment

Using pip - install each of the above

Download and unzip the motion forecasting dataset here: https://www.argoverse.org/av2.html#download-link

The examples need to be preprocessed, you can either load each example with the AV2.process_parallel_frenet function, or uncomment and run the following section at the bottom of the dataset.py file to save the preprocessed files and retrieve them with __getitem__, see [dataset.py](#datasetpy) for important details. Also uncomment the np.save calls in the function. The process_parallel_frenet function is somewhat slow (~0.2 sec) so if you're going to run things repeatedly, it's worth it to save the preprocessed files. You can exclude the map over x or y, which are the train and val datasets if you only want to process one.

``` python
with mp.Pool() as pool:
    pool.map(x.process_parallel_frenet, range(len(x)))
    pool.map(y.process_parallel_frenet, range(len(y)))
    pool.close()
    pool.join()
```

## pcurvenet.py
Houses the training script and additional utility objects to compute forward passes. Run the main function to train the model. Run validate to compute some of the ADE metrics. Also used to call the metrics functions (iar = illegal action rate). 

### pcurve_checkpoints
The v0 directory just holds backup copies, currently the last.ckpt checkpoint is the best performing, use it for metrics. last-v2.ckpt is a not fully trained probabilistic version (using cross entropy instead of binary cross entropy). It is not as good, but might be improvable.

## data_utils.py
Mostly houses functions used by the dataset for preprocessing. Closer to the bottom though, there are a few functions used for translating (s, d) coordinates in the frenet frame to (x, y) coordinates. See https://github.com/fjp/frenet for an explanation about the frenet frame. Angle between is an important function that calculates the positive [0, 2pi], smallest angle between two angles. 

## dataset.py
For pcurvenet, process_parallel_frenet is the preprocessing function. __getitem__ loads preprocessed items and additionally loads the ego_idx and the normalizing coordinates (mins).

## data_handling.py
Houses initial versions for functions and testing, these are unimportant. The pcurve function though, which is currently being called, can be used to visualize scenarios. 

