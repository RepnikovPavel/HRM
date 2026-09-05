create a data layer for the current HRM task.
- docker files and scripts for data download
- identify the data storage paths(on the dev machine you can use /mnt/nvme, on server you can use /mnt/hdd2)
- visualization functionality of a specific dataset or simulation. The data for visa processing should be stored next to the data for training or simulation.
- The Internet is limited on the developer's machine,unlimited Internet on the server, so it is more important to download on the server, and then do rsync over the local network. there may also be problems with downloading due to the restrictions of providers or the regulator, so these restrictions may lead to errors in the download code, this code must be written precisely and variably for different network settings.  

move submodules for datasets to embedded code to repo.  

```
[submodule "dataset/raw-data/ConceptARC"]
	path = dataset/raw-data/ConceptARC
	url = git@github.com:victorvikram/ConceptARC.git
[submodule "dataset/raw-data/ARC-AGI"]
	path = dataset/raw-data/ARC-AGI
	url = git@github.com:fchollet/ARC-AGI.git
[submodule "dataset/raw-data/ARC-AGI-2"]
	path = dataset/raw-data/ARC-AGI-2
	url = git@github.com:arcprize/ARC-AGI-2.git
```

tidy up the data preprocessing layer, add containers, get the paths to the files on the disk in the local setup.
```
# Initialize submodules
git submodule update --init --recursive

# ARC-1
python dataset/build_arc_dataset.py  # ARC offical + ConceptARC, 960 examples
# ARC-2
python dataset/build_arc_dataset.py --dataset-dirs dataset/raw-data/ARC-AGI-2/data --output-dir data/arc-2-aug-1000  # ARC-2 official, 1120 examples

# Sudoku-Extreme
python dataset/build_sudoku_dataset.py  # Full version
python dataset/build_sudoku_dataset.py --output-dir data/sudoku-extreme-1k-aug-1000  --subsample-size 1000 --num-aug 1000  # 1000 examples

# Maze
python dataset/build_maze_dataset.py  # 1000 examples

```