import os
import subprocess


""" NOTE
For kgtk to be recognized as a command, set an environment variable = "kgtk" and use the environment variable
isntead of using "kgtk" directly in your command. 
e.g.:
os.environ['kgtk'] = "kgtk"
run_command("$kgtk query etc...
"""
def run_command(command, substitution_dictionary = {}):
    """Run a templetized command."""
    cmd = command
    for k, v in substitution_dictionary.items():
        cmd = cmd.replace(k, v)
    
    # print(cmd)
    output = subprocess.run([cmd], shell=True, universal_newlines=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if output.stdout:
        print(output.stdout)
    if output.stderr:
        print(output.stderr)

"""
example of using run_command:

type_mapping_file = "../../data/Q44_profiler_output/Q44.type_mapping.tsv"
command = "$kgtk query -i TYPE_MAPPING_FILE -i $DATA/$NAME.label.en.tsv --graph-cache $STORE \
          --match '`TYPE_MAPPING_FILE`: (n1)-[]->(type), label: (type)-[:label]->(lab)' \
          --return 'distinct type as type, lab as type_label, count(distinct n1) as count, \"_\" as id' \
          --where 'lab.kgtk_lqstring_lang_suffix = \"en\"' \
          --order-by 'count(distinct n1) desc' \
          --limit 5 \
          | column -t -s $'\t' \
          && ls"
run_command(command, {"TYPE_MAPPING_FILE" : os.path.abspath(type_mapping_file)})
"""

def rename_cols_and_overwrite_id(file_path_wo_ext, ext, old_cols_str, new_cols_str):
    command = "$kgtk rename-columns -i {0}{1} -o {0}_temp{1} \
               --old-columns {2} --new-columns {3} \
               && $kgtk add-id -i {0}_temp{1} \
               -o {0}{1} --overwrite-id \
               && rm {0}_temp{1}".format(file_path_wo_ext, ext, old_cols_str, new_cols_str)
    run_command(command)