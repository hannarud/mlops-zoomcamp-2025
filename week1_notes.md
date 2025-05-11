# Week 1 notes

1. Important thing to consider during the first stage (design): do we actually need Machine Learning to solve the problem? This is not necessarily the case!!!
2. After running Codespaces locally in my VSC: I'll be using the Python available (version 3.12 is available there, I won't install Conda), but use `virtualenv` to operate environments.
  Just in case I'll need to repeat it, here is the chain of commands:
    * `ls` - working dir should be the current repo root dir
    * `python -m venv mlops-zoomcamp-env`
    * `source mlops-zoomcamp-env/bin/activate`
    * `pip list` will most likely show that pip can be updated
    * `pip install --upgrade pip`
3. Also, need to install Jupter Notebook. So, starting the [reqirements.txt file](requirements.txt). And adding packages there as I discover I need them for the homeworks etc. In order to install, use `pip install -r requirements.txt`.
4. See not much sensein using codespaces now, continuing locally, at least for a while.
5. Installing packages straight from the .ipynb: `!pip install pandas seaborn scikit-learn`. Ofthen need to restart the kernel afterwards in order for the changes to apply.
6. Downloading the data straight from Jupyter using `!wget` and specifying output file with `-O` option.
7. For storing Jupyter Notebooks in git, it might make sense to strip notebook output. How to do it: https://stackoverflow.com/a/73218382. In this case, notebooks with executed cells won't get into git, which is nice from one side (no heavy overwriting on each notebook re-execution), but is bad on another side (we don't preserve the actually executed notebooks and are not able to see results). I still prefer to keep notebbooks in git tidy and for storing the results of execution use export to .html + pu the to `results` folder, and add this folder to `.gitignore`. I then use .html files to share them with stakeholders when needed by uploading them to Google Drive or similar (i.e. not version control).
