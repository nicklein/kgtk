import pandas as pd
from type_mapping import build_type_mapping


if __name__ == "__main__":
    
    type_mapping = build_type_mapping()
    quantity_df = pd.read_csv("Q44.part.quantity.tsv", "\t")
    
    for index, row in quantity_df.iterrows():
        entity = row["node1"]
        prop = row["label"]
        value = row["node2"]
        
        # Add this <property, value> pair to the label sets corresponding to this entity's types
        if entity not in type_mapping:
            continue
        for entity_type in type_mapping[entity]:
            #TODO
    
    