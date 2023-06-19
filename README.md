# Transformer for Compositional Generalization

## Preparation
1. Clone this repo.
```shell
git clone git@github.com:liqing-ustc/TransformerCoT.git
```
2. Set up the environment using conda. This repo is currently developed on Pytorch 2.0.1.
```shell
conda env create -f conda-environment.yaml
```

## Running
```shell
python run.py [--config-name <CONFIG_NAME>] [override parameters]
```

<!-- 
Before running, make sure your config file has been properly set up. An example config file can be found in ```configs```.
Then to run the file, use ```launch.py``` with proper arguments.
```shell
# python launch
python launch.py --config <PATH_TO_CONFIG_FILE> <ADDITIONAL_HYDRA_CONFIG>

# accelerate launch
python launch.py  --launch_mode accelerate --config <PATH_TO_CONFIG_FILE> <ADDITIONAL_HYDRA_CONFIG>

# SLURM submitit launch
python launch.py --launch_mode submitit --config <PATH_TO_CONFIG_FILE> <ADDITIONAL_HYDRA_CONFIG>
```
For more arguments, use ```python launch.py --help``` for more key word argument info about 
[SLURM submitit](https://github.com/facebookincubator/submitit) launch or 
[Accelerate](https://huggingface.co/docs/accelerate/index)
launch settings. We also provide more example launching commands in the following sections. 

## General design guidance
In the current version, you will most likely modify files in the following directories to add a new model pipeline to the code base.
Suppose that you have a model ```MODEL``` with new modules ```MODULE```, new optimization objectives ```LOSS```, new training logic ```TRAINER```, and new task evaluation metrics ```EVAL```, you will need to create/modify files in the following directory:
```
|-- modules/xxx/MODULE.py   # depending on whether a module can be categorized into vision/language/head/third_party
|-- model/MODEL.py          # for specifying the correct forward pipeline using helper modules and setting optimization strategies (e.g., decay, lr)
|-- optim/loss.py           # add your new LOSS
|-- trainer/TRAINER.py      # define train/eval/test logic for running training or inference
|-- evaluator/EVAL.py       # define the new evaluation metric after obtaining training/inference results
```

After creating these files, to correct configure the experiments, you will need to add your initialization arguments/hyperparameters for all newly created classes. In ```configs/default.py```, we provide an example to initialize all modules in Vista3D. The general design choice in configuring models is to let the configuration keywords align with initialization arguments to avoid errors. So in ```configs/default.py``` you should see ```model.name``` for knowing the current model choice and ```model.args``` for all arguments initializing that model. This could be generalized to all object initialization (temporarily, this logic could be changed based on discussions).

If you want to dive deep into how existing modules/models/pipelines are implemented, the basic logic is to have a ```build.py``` for each base object class, implementing general init/utility functions, and/or leaving interface functions that must be implemented in child classes. Then in each folder, you will further see object classes with their unique signature (e.g. ```vista3d.py```, ```grounding_head.py```, ```vista3d_trainer.py```) defining all customized class-depenedent features (temporarily, this logic could be changed based on discussions).
-->

## Additional note
1. As we currently use ```Registry``` from ```fvcore.common.registry```, when adding new modules to each folder including
   (```model```, ```modules```, ```trainer```, ```evaluator```) and all other directories
    that contain a ```build.py``` file for accessing registered classes, remember to properly add the
    the dependency in the corresponding ```__init__.py``` file so that the classes are registered before importing.
    For example:
	```python
	# add to model/__init__.py
	from .my_model import *
	
	# add to modules/vision/__init__.py
	from .my_vision_module import *
	
	# add to evaluator/__init__.py
	from .my_evaluator import *
	
	# add to trainer/__init__.py
	from .my_trainer import *
    ```
2. During implementation, make sure that you have implemented all base class functions. And if you feel that a function could be upgraded
   to be a base class function (at least on your local branch), do remember to first implement interface functions functions in base classes
   so that it can help you remember to overwrite this function in each customized child class. On the main branch, do remember to provide unit
   test cases for code reviews.

3. To log on your weight and bias account, when remember to initialize through
    ```shell
    # Initialize weight and bias through your account
    wandb init
    # Follow instruction following this command to sign in
    ...
    # run command
    python run.py 
    ```