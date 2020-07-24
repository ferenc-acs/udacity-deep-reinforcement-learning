# Pong Reinforce
## Attention!

This notebook requires another environment and currently (2020-07-24) run **only** on the Udacity Cloud!

Reasons are roughly:

* Executing the notebook in the standard environment for this repository results in a ```AttributeError: 'HTMLWriter' object has no attribute '_temp_names'``` 
* This is because JSAnimation requires Matplotlib 2.1.0
* Matplotlib can not be downgraded to 2.1.0 on my machine because this would collide with the present CUDA and Python installation:
* Furthermore [JSAnimation is deprecated](https://github.com/jakevdp/JSAnimation) and not further maintained.

This is especially frustrating because doing this exercise properly requires to do training runs that require around one hour training time on average hardware. On the Udacity cloud this means burning valuable GPU hours.

If you want to tinker around with creating an enviroment, see ```requirements.txt``` in this directory, that works on a modern machine or are open to code refactoring I would highly appreciate it! **\*SCNR***

### Error when attempting to downgrade Matplotlib to 2.1.0

<pre>(drlnd) <font color="#4E9A06"><b>ferenc@Anubis-Linux</b></font>:<font color="#3465A4"><b>~/Python/rl/udadrl/udacity-deep-reinforcement-learning/pong1-reinforce</b></font>$ conda install matplotlib=2.1.0 --dry-run
Collecting package metadata (current_repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Collecting package metadata (repodata.json): done
Solving environment: failed with initial frozen solve. Retrying with flexible solve.
Solving environment: - 
Found conflicts! Looking for incompatible packages.
This can take several minutes.  Press CTRL-C to abort.
failed                                                                                                                                                                              

UnsatisfiableError: The following specifications were found
to be incompatible with the existing python installation in your environment:

Specifications:

  - matplotlib=2.1.0 -&gt; python[version=&apos;&gt;=3.7,&lt;3.8.0a0|&gt;=3.8,&lt;3.9.0a0&apos;]

Your python: python==3.6.10=h7579374_2

If python is on the left-most side of the chain, that&apos;s the version you&apos;ve asked for.
When python appears to the right, that indicates that the thing on the left is somehow
not available for the python version you are constrained to. Note that conda will not
change your python version to a different minor version unless you explicitly specify
that.

The following specifications were found to be incompatible with your CUDA driver:

  - feature:/linux-64::__cuda==10.2=0

Your installed CUDA driver is: 10.2


(drlnd) <font color="#4E9A06"><b>ferenc@Anubis-Linux</b></font>:<font color="#3465A4"><b>~/Python/rl/udadrl/udacity-deep-reinforcement-learning/pong1-reinforce</b></font>$ 
</pre>