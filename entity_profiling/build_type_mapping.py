import os
import pandas as pd


def build_type_mapping():
    # create edge file
    os.system("kgtk filter -i Q44/Q44.part.wikibase-item.tsv -o entity_types.tsv --pattern ' ; P31 ; '")
    
    entity_types_df = pd.read_csv("entity_types.tsv", sep = "\t")
    entity_types_df.drop(columns = ["id","label"], inplace = True)
    entity_types_df = entity_types_df.groupby("node1")["node2"].apply(list)
    entity_types_df = entity_types_df.T
    return entity_types_df.to_dict()
    
if __name__ == "__main__":
    type_mapping = build_type_mapping()
    print(type_mapping["Q1011"])