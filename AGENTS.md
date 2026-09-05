## source paper  

the original text of the HRM paper is attached in the file paper/document.md

## original repo github issues and pull requests  

folder github_activity  


## basics of programming

1. run the code only from under the docker container.
2. use the C programming style, no matter what language you program in.
3. any dependency used in the repository must be explicitly specified in the docker file.  
4. you can't write comments in the code.  
5. Avoid boilerplate code, use guile rather than templates. it's better to write fewer lines of code than to create a bunch of classes and structures.  
6. avoid inheriting classes with a depth of more than 1.  
7. measure the code execution time. it almost always turns out that the time spent on optimization is less than the waiting time for execution of a slow, naively written program, even taking into account multiple restarts. a good instructive lesson was presented in the work of modded-nanogpt HRM/examples/modded-nanogpt/README.md  
8. You must use 100% of the calculator's resources.  all RAM, all CPU resources, all GPU resources, all disk resources, because the main thing is to minimize the time for performing calculations.  
9. for each stage of the calculations, it is necessary to make an approximate calculation of the estimated cost of running the same task, but in the cloud. A formula and a starting number of $ must be obtained. simply put, it is necessary to keep accounting calculations in separate documents.  
10. It's more important to spend time planning code writing than writing code, as mindless code writing usually leads to a zugzwang from which there is no way out except to start from the beginning. that is, every time you need to rethink yourself, okay, I'll write this function or file, and then what will I do with this code?  
11. it is necessary to maintain similar docker files for at least two architectures in parallel, for both cuda 12 and cuda 13, since the code will run not only on the development machine, but also on the server, where there may be a different environment. for the same reason, when writing pytorch code, it is necessary to support not only single gpu scenarios, but also multi gpu scenarios.  
12. development should be library driven, prefer the explicit creation of packages/libraries/extensions rather than simply storing files in folders. the code must be compiled and installed using standard tools such as pip, cmake, and so on. for the same reason, it is forbidden to import dependencies during code execution, inside functions.  from the point of view of c++, compilation in runtime is prohibited, from the point of view of python, dynamic import of modules is prohibited.
13. if something goes wrong and the task does not respond, then it is important to pause and stop completing tasks, most likely the problem is in the formulation. let's say a user asked to download a file, read it, and then solve the quicksort problem described in the file. if you couldn't download the file, then there's no point in making a rollback to the quick sort solution in the general formulation, perhaps there was critical information in the file that completely changes the approach to solving the problem.  
14. you cannot store data in a repository, the data must be stored elsewhere, not in code.  
15. For each dataset or simulation environment under consideration, it is necessary to create a visual demo and detailed descriptions of the data/simulation. The demo must be run from the command line and contain graphics. it is advisable to create md/html documents with a summary of the dataset, for example, the amount of disk space, the number of examples, the variability of the environment, the size of the vectors of observations of the agent or the state of the environment, the size of the vector of actions, the number of tokens in the dataset or the number of examples in the dataset, the number of rows in the dataset, the number of columns in the dataset, and so on. prefer graphics in the browser rather than desktop. prefer the architecture of micro frontends when writing applications in the browser. 
16. you should not trust other people's work, articles, programs, time has shown that even the most quoted works contain offensive errors.  


## Code and commit style

Observed conventions on branch `deepbench` — follow them:

- **English everywhere** in code, comments, docs and commit messages.
- **One docker image per stage.** Wrapper scripts in `scripts/*.sh` build
  their own image from their own `docker/DockerFile<Stage>` and run it with
  `--rm --init --user $(id -u):$(id -g)`, mounting only what the stage needs.
  Never add dependencies to an existing stage's image for a new purpose —
  create a new Dockerfile instead.
- **Comments explain why, not what**; tricky semantics (overflows, resume
  rules, dtype bounds) get a paragraph with the measured/verified fact.
- **Claims are measured.** Docs and commit messages carry real numbers
  ("93.2M tok/s", "byte-identical", "99.80% vocab overlap"). Never state a
  performance or parity claim you did not verify on this machine.
- **Commit messages**: one-line subject, dense, with the outcome/numbers in
  parentheses; no conventional-commits prefixes, no body unless needed.
  Examples: `count_tokens: 31.5->23.3 min (flat-cache + readahead + 256MiB
  batches), verified same 176.1B tokens`.
- **Docs next to code**: each subsystem has `docs/` with pipeline notes and a
  `gotchas.md`-style file recording known issues, including deliberately
  preserved bug-for-bug parity.
- Generated docs (e.g. `scripts/docs/*.md`) are produced by their generator
  scripts — patch the generator template *and* the generated file.

## Hard rules

- **No `git commit`/`git push` without an explicit go-ahead** in the current
  conversation. Prepare the change, show it, wait.
- **Do not rebuild/replace existing docker images** for a different purpose;
  new stage → new image name.
- Long-running stage commands run in foreground streaming logs (scripts use
  `python -u` / `PYTHONUNBUFFERED=1`); indicatif/tqdm bars are invisible in
  `docker logs`, so stages print explicit progress lines — keep that pattern.
