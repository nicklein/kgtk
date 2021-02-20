import os
import sys
import papermill as pm

#TODO
# - add file validation and print out out params, prompt to verify they are right before running

#=================================================#
#                   Parameters                    #
#=================================================#

"""
File and directory params that you likely need to change if you
want to run the profiler on a new dataset:
    profiler_run_dir: Directory to save output files and notebooks to
"""
data_dir = "./data/wikidata_humans" # my data files are all in the same directory, so I'll reuse this path prefix
untrimmed_quantity_file_in = "{}/claims.quantity.tsv.gz".format(data_dir)
quantity_qualifiers_file = "{}/qualifiers.quantity.tsv.gz".format(data_dir)
trimmed_quantity_file_out = "{}/claims.quantity_trimmed.tsv.gz".format(data_dir)
item_file = "{}/claims.wikibase-item.tsv.gz".format(data_dir)
time_file = "{}/claims.time.tsv.gz".format(data_dir)
quantity_file = "{}/claims.quantity_trimmed.tsv.gz".format(data_dir)
label_file = "{}/labels.en.tsv.gz".format(data_dir)
work_dir_name = "wikidata_humans"
type_to_profile = "Q5" # Human
string_file = None #"{}/claims.string.tsv.gz".format(data_dir)

"""
Params that don't *need* to be changed every time you run on a new dataset...
These will affect the results, but don't need to be changed for the profiler
to run correctly on your dataset:
"""
# HAS_embeddings notebook
# Embedding model params
directed = False
num_H_walks = 10
num_A_walks = 0 # TODO - change back to 10 once address scalability issue
num_S_walks = 0 # TODO - change back to 10 once address scalability issue
walk_length = 10
representation_size = 64
window_size = 5
workers = 32
k = 5
# File/Directory params
h_walks_filename = "h_walks.txt"
a_walks_filename = "a_walks.txt"
s_walks_filename = "s_walks.txt"

"""
Flags to pick and choose which notebooks to run.
"""
# run_quantity_trim = True
# run_explore_ents = True
# run_label_creation = True
# run_filter = True
run_quantity_trim = False
run_explore_ents = False
run_label_creation = False
run_filter = False

run_embeddings = True
run_select_label_set = True
run_view_profiles = True

#=================================================#
#                   End of Parameters             #
#=================================================#


# Ensure all paths are absolute. While the notebooks will do this processing
# as well to handle them being run without this script, we need to resolve
# relative paths now since output notebooks may be saved in a different directory.
untrimmed_quantity_file_in = os.path.abspath(untrimmed_quantity_file_in)
quantity_qualifiers_file = os.path.abspath(quantity_qualifiers_file)
trimmed_quantity_file_out = os.path.abspath(trimmed_quantity_file_out)
item_file = os.path.abspath(item_file)
time_file = os.path.abspath(time_file)
quantity_file = os.path.abspath(quantity_file)
label_file = os.path.abspath(label_file)
if string_file:
    string_file = os.path.abspath(string_file)
# Set work and store directory paths
work_dir = os.path.abspath("./output/{}".format(work_dir_name))
store_dir = os.path.abspath("{}/temp".format(work_dir))
    
# Validation
err_msg = None
if not os.path.isfile(untrimmed_quantity_file_in):
    err_msg = "No such file for untrimmed_quantity_file_in: {}".format(untrimmed_quantity_file_in)
elif not os.path.isfile(quantity_qualifiers_file):
    err_msg = "No such file for quantity_qualifiers_file: {}".format(quantity_qualifiers_file)
elif not os.path.isfile(quantity_file) and not (quantity_file == trimmed_quantity_file_out and run_quantity_trim):
    err_msg = "quantity_file ({}) does not exist, nor will it be created by the trim_quantity_file notebook".format(quantity_file)
elif not os.path.isfile(item_file):
    err_msg = "No such file for item_file: {}".format(item_file)
elif not os.path.isfile(time_file):
    err_msg = "No such file for time_file: {}".format(time_file)
elif not os.path.isfile(label_file):
    err_msg = "No such file for label_file: {}".format(label_file)
elif not os.path.isfile(untrimmed_quantity_file_in):
    err_msg = "No such file for untrimmed_quantity_file_in: {}".format(untrimmed_quantity_file_in)
elif string_file and not os.path.isfile(string_file):
    err_msg = "A string_file was specified, but no such file exists: {}".format(string_file)
if err_msg:
    sys.exit(err_msg)
    
# Create directory for papermill output notebooks
pm_out_dir = "{}/pm_notebooks".format(work_dir)
if not os.path.exists(pm_out_dir):
    os.makedirs(pm_out_dir)
    
print("Output of notebooks will be saved under {}".format(work_dir))
print("The output notebooks will be saved here: {}".format(pm_out_dir))

# Run notebooks
if run_quantity_trim:
    print("Running 0_trim_quantity_file.ipynb")
    pm.execute_notebook(
        "0_trim_quantity_file.ipynb",
        "{}/0_trim_quantity_file.out.ipynb".format(pm_out_dir),
        parameters=dict(
            quantity_file = untrimmed_quantity_file_in,
            qualifiers_file = quantity_qualifiers_file,
            out_file = trimmed_quantity_file_out,
            work_dir = work_dir,
            store_dir = store_dir
        )
    )
if run_explore_ents:
    print("Running explore_entities.ipynb")
    pm.execute_notebook(
        "explore_entities.ipynb",
        "{}/explore_entities.out.ipynb".format(pm_out_dir),
        parameters=dict(
            item_file = item_file,
            time_file = time_file,
            quantity_file = quantity_file,
            label_file = label_file,
            work_dir = work_dir,
            store_dir = store_dir,
            string_file = string_file
        )
    )
if run_label_creation:
    print("Running 1_candidate_label_creation.ipynb")
    pm.execute_notebook(
        "1_candidate_label_creation.ipynb",
        "{}/1_candidate_label_creation.out.ipynb".format(pm_out_dir),
        parameters=dict(
            item_file = item_file,
            time_file = time_file,
            quantity_file = quantity_file,
            label_file = label_file,
            work_dir = work_dir,
            store_dir = store_dir,
            string_file = string_file
        )
    )
if run_filter:
    print("Running 3_candidate_filter.ipynb")
    pm.execute_notebook(
        "3_candidate_filter.ipynb",
        "{}/3_candidate_filter.out.ipynb".format(pm_out_dir),
        parameters=dict(
            work_dir = work_dir,
            store_dir = store_dir
        )
    )
if run_embeddings:
    print("Running 5_HAS_entity_embeddings.ipynb")
    pm.execute_notebook(
        "5_HAS_entity_embeddings.ipynb",
        "{}/5_HAS_entity_embeddings.out.ipynb".format(pm_out_dir),
        parameters=dict(
            item_file = item_file,
            time_file = time_file,
            quantity_file = quantity_file,
            label_file = label_file,
            work_dir = work_dir,
            store_dir = store_dir,
            directed = directed,
            num_H_walks = num_H_walks,
            num_A_walks = num_A_walks,
            num_S_walks = num_S_walks,
            walk_length = walk_length,
            representation_size = representation_size,
            window_size = window_size,
            workers = workers,
            type_to_profile = type_to_profile,
            k = k,
            h_walks_filename = h_walks_filename,
            a_walks_filename = a_walks_filename,
            s_walks_filename = s_walks_filename
        )
    )
if run_select_label_set:
    print("Running 6_select_final_label_set.ipynb")
    pm.execute_notebook(
        "6_select_final_label_set.ipynb",
        "{}/6_select_final_label_set.out.ipynb".format(pm_out_dir),
        parameters=dict(
            item_file = item_file,
            time_file = time_file,
            quantity_file = quantity_file,
            label_file = label_file,
            work_dir = work_dir,
            store_dir = store_dir,
            string_file = string_file,
            type_to_profile = type_to_profile
        )
    )
if run_view_profiles:
    print("Running 7_view_profiles.ipynb")
    pm.execute_notebook(
        "7_view_profiles.ipynb",
        "{}/7_view_profiles.out.ipynb".format(pm_out_dir),
        parameters=dict(
            work_dir = work_dir,
            store_dir = store_dir,
            entity_type = type_to_profile
        )
    )
    