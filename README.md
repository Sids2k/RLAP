# RLAP
Unsupervised Learning of Neuro-symbolic Rules for Generalizable Context-aware Planning in Object Arrangement Tasks

## Scripts
### The pipeline to use them is:
1. dataGen.py
2. modelLearner.py
3. pmodel.py
4. planner.py

The mcts.py is a helper script only and needn't be invoked directly.

### Short note for the scripts
1. All paths are not updated in the script, and should be edited before use.
2. Some scripts have variables hardcoded and others take them as parameters, please change them in their respective ways, if needed, before use.

## Simulation
### Pipeline to run simulation
1. To generate the curriculum.json file, we need to call plan2json.py which makes a json file for the plan and updates curriculum.json
2. construct_dataset.py is the main script to call once the curriculum.json file has been generated to see the simulation
