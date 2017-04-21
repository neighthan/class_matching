# hkn_tutor
Tutor/tutee matching code for HKN at MIT.

There are two files of note here: the notebook and the script.

The notebook (tutor_tutee_matching.ipynb) shouldn't necessarily be used to do the actual matching, but it is very useful for experimentation with changes, viewing intermediate results, and further documentation of the code. You'll find a few cells there that explain the approach being taken to match the tutors and tutees in much more detail than the comments / docstrings in the script.
Additionally, the notebook contains a function to generate sample data for testing and a function to perform basic analysis on the matchings generated.

The script should be used when you need to do the matching itself. I've provided what I think are sane defaults for all of the parameters, so you can just run it with `/path/to/matching.py` (if you have /usr/bin/python) or `python /path/to/matching.py` more generally.
Run `matching.py -h` to see the parameters you can set as well as explanations. In brief:
- `-v` or `--verbose` to print much more information while running
- `-lc` (lambda_classes) and `-ls` (lambda_students) to specify how much extra weight should be given to priority classes (those with a larger ratio of tutees to tutors) and priority students (those with a class marked as priority in their list), respectively. One can specify a list of lambda values, and one matching will be run for each pair. In my experiments, the weights don't cause *major* changes, but they can make a difference. They seem a bit finicky, I'm afraid, so it's probably worth trying a few values and then picking the best matching. You can specify multiple weights like: `matching.py -lc 1 2 5 -ls 1 5 10`.
- `-prod` or `--cartesian_product` if you want to do one matching for each combination of lambdas. For example, running `matching.py -lc 1 2 -ls 1 5` (without `--prod`) would compute two matchings: one with the lambdas being (1, 1) and the other with (2, 5). However, running `matching.py -lc 1 2 -ls 1 5 --prod` will compute four matchings: one each with lambdas (1, 1), (1, 5), (2, 1), (2, 5).
