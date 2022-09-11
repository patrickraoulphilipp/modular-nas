Modular Neural Architecture Search
===
Keras implementation of *Towards Modular Neural Architecture Search* (using tf.compat.v1 for now, but tf2 will be supported soon). The NAS approach uses NASBench to sample possible network architectures and provides a transformation function of tf1 models to keras. Two straghtforward weight sharing mechanisms (weight reuse and weight averging based on a sliding window) are integrated as well to speed up the search process.

Instructions
-------------
1.  Clone this repo.

```
git clone https://github.com/patrickraoulphilipp/modular-nas
cd modular-nas
```

2. (optional) Create a virtualenv. The implementation has been tested for Python 3.9.

```
virtualenv venv
source venv/bin/activate
```

3. Install all dependencies (including NASBench version for tf.compat.v1) for cpu or gpu usage.

```
pip install -r requirements_{cpu|gpu}.txt .
```

4. Download NASbench tfrecord file with model evaluations (https://github.com/google-research/nasbench#Download%20the%20dataset).

```
Direct link for small dataset for 108 epochs: https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord
```

5. Set parameters in config.py, including dataset (cifar10, cifar100, mnist or fashion-mnist for now), path to NASBench tfrecord file and NAS-specfic parameters. Parameter DEBUG_EVOLUTION is used to skip model trainings by replacing the standard fitness function (accuracy on validation data) with the number of modules in the chosen architecture. This only supports debugging the evolutionary algorithm and correct keras model transformation.

```
DEBUG_EVOLUTION = False
PATH_TO_TFRECORD = "/PATH/TO/nasbench_only108.tfrecord"
eval_data_set = "cifar10"
...
```

6. Run main.py to start the search process.

```
python main.py
```

Cite as
-------------
	@inproceedings{boecking2020,
		author    = {Lars B\"ocking and Patrick Philipp and Cedric Kulbach},
		title     = {Towards Modular Neural Architecture Search},
		booktitle = {Proceedings of the 1st Workshop on Neural Architecture Search co-located with the 8th International Conference on Learning Representations, ICLR\textquotesingle20},
		year      = {2020},
	}