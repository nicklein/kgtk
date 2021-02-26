import pandas as pd
import json


class ProfileRetriever:
    
    def __init__(self, labels_to_entities_file, ordered_label_set_file):
        labels_df = pd.read_csv(labels_to_entities_file, delimiter = '\t').fillna("")
        # dictionary of label_id --> set of matching entities
        # using groupby is not the most efficient, but more concise
        self.label_to_entities = labels_df.groupby("node1")["node2"].apply(set).to_dict()
        self.profiling_type = labels_df.loc[0,"type"]
        self.profiling_type_label = labels_df.loc[0,"type_label"]
        info_cols = ["node1", "type", "type_label", "prop", "prop_label", "value", "value_label", "value_lb", "value_ub", "prop2", "prop2_label", "value2", "value2_lb", "value2_ub", "si_units", "wd_units"]
        self.label_info = labels_df.loc[:,info_cols].rename(columns={"node1":"label_id"}).groupby("label_id").first()
        
        with open(ordered_label_set_file, 'r') as f:
            self.label_set = json.load(f)
            
    def get_entity_profiles(self, entities, max_labels_in_profile=5):
        return [self.get_entity_profile(e, max_labels_in_profile) for e in entities]
    
    def get_entity_profile(self, entity, max_labels_in_profile=5):
        profile = []
        for l in self.label_set:
            if len(profile) >= max_labels_in_profile:
                break
            if entity in self.label_to_entities[l]:
                profile.append(l)
        return profile
    
    def get_label_info(self, label_id):
        return self.label_info.loc[self.label_info["label_id"] == label_id]
    
    def get_label_set_size(self):
        return len(self.label_set)
    def get_profiling_type(self):
        return profiling_type
    def get_profiling_type_label(self):
        return profiling_type_label