

# loading data from table
def get_candidates_by_cell(table_df):
    cells = df.groupby("row")["kg_id"].apply(list).to_list()
    return cells
def get_cell_ground_truths(table_df):
    cells_gt = [l[0] for l in df.groupby("row")["GT_kg_id"].apply(list).to_list()]
    return cells_gt
# loading type mapping
def load_type_mapping(type_mapping_file):
    df = pd.read_csv(type_mapping_file, delimiter='\t')
    type_mapping = df.groupby("node1")["node2"].apply(list).to_dict()
    return type_mapping
# retrieving profiles
def get_entity_profile(profile_dict, entity, max_labels_in_profile = 5):
    return profile_dict[entity][:max_labels_in_profile] if entity in profile_dict else []
def get_entity_profiles(profile_dict, entities, max_labels_in_profile = 5):
    return [get_entity_profile(profile_dict, ent, max_labels_in_profile) for ent in entities]

# getting table info and coverage of profiles and embeddings
def get_gt_cands(cells, cells_gt):
    return [cells_gt[i] for i in range(len(cells_gt)) if cells_gt[i] in cells[i]]
def get_num_cands_w_profiles(cells, profile_dict):
    return sum([1 for cell in cells for cand in cell if cand in profile_dict])
def get_num_gt_cands_w_profiles(gt_cands, profile_dict):
    return sum([1 for gt_cand in gt_cands if gt_cand in profile_dict])
def get_num_cands_w_embeddings(cells, embeddings):
    return sum([1 for cell in cells for cand in cell if cand in embeddings])
def get_num_gt_cands_w_embeddings(cells, embeddings):
    return sum([1 for gt_cand in gt_cands if gt_cand in embeddings])


#=============================
# Scoring Methods
#=============================

# Random
def random_scoring(cells):
    # is it more helpful to see the scores for each, or the final choice? I think the former.
    scores=[]
    for cell in cells:
#         choice = random.choice(cell)
#         scores.append({candidate : 1 if candidate == choice else 0 for candidate in cell})
        scores.append({candidate : 1/len(cell) for candidate in cell})
    return scores

# Profile methods
def random_amongst_profile_ents(cells, profile_dict):
    candidate_probs = []
    for cell_idx, candidates in enumerate(cells):
        cell_cand_probs = {cand : 1 if cand in profile_dict else 0 for cand in candidates}
        # normalize within each cell
        denominator = sum(cell_cand_probs.values())
        for cand in cell_cand_probs:
            cell_cand_probs[cand] /= denominator
        candidate_probs.append(cell_cand_probs)
    
    return candidate_probs

def avg_cell_profile_intersect_size_candidate_scoring(cells, profile_dict, max_labels_in_profile=5):
    #TODO - wow this code is ugly. come back to this when not so tired.
    cell_candidate_profiles = [get_entity_profiles(profile_dict, candidates, max_labels_in_profile) for candidates in cells]
    cell_candidate_profile_overlaps = []
    
    # First build up dictionaries of label counts so we can do this computation linearly
    label_counts_in_table = {}
    label_counts_by_cell = {}
    for cell_idx in range(len(cells)):
        if cell_idx not in label_counts_by_cell:
            label_counts_by_cell[cell_idx] = {}
        for profile in cell_candidate_profiles[cell_idx]:
            for label_id in profile:
                if label_id not in label_counts_in_table:
                    label_counts_in_table[label_id] = 0
                if label_id not in label_counts_by_cell[cell_idx]:
                    label_counts_by_cell[cell_idx][label_id] = 0
                label_counts_in_table[label_id] += 1
                label_counts_by_cell[cell_idx][label_id] += 1
                
    # For each candidate, find number of labels in other cell's candidate's profiles that match
    # a label in this candidate's profile.
    for cell_idx in range(len(cells)):
        candidate_profile_overlaps = []
        for profile in cell_candidate_profiles[cell_idx]:        
            overlap = 0
            for label_id in profile:
                overlap += (label_counts_in_table[label_id] - label_counts_by_cell[cell_idx][label_id])
            candidate_profile_overlaps.append(overlap)
        cell_candidate_profile_overlaps.append(candidate_profile_overlaps)
    # Normalize counts for candidates within each cell
    cell_candidate_profile_overlaps = [[overlap / sum(overlaps) for overlap in overlaps] for overlaps in cell_candidate_profile_overlaps]
    
    # put in list of dictionaries format (one {candidate : score} dictionary per cell)
    cell_candidate_scores = [{cells[i][j] : cell_candidate_profile_overlaps[i][j] for j in range(len(cells[i]))} for i in range(len(cells))]
    
    return cell_candidate_scores

def max_cell_profile_intersect_size_candidate_scoring(cells, profile_dict, max_labels_in_profile=5):
    profiles = {cand : set(get_entity_profile(profile_dict, cand, max_labels_in_profile)) for candidates in cells for cand in candidates}
    cells_candidate_profile_overlaps = []
    for cur_cell_idx, candidates in enumerate(cells):
        cell_candidate_profile_overlaps = {}
        for cand in candidates:
            cand_profile = profiles[cand]
            
            if len(cand_profile) == 0:
                cell_candidate_profile_overlaps[cand] = 0
                continue
                
            cumulative_max_per_cell_overlap = 0
            for cell_idx in range(len(cells)):
                # Only compare to candidates in other cells
                if cell_idx == cur_cell_idx:
                    continue
                cumulative_max_per_cell_overlap += max([len(cand_profile & profiles[other_cell_cand]) for other_cell_cand in cells[cell_idx]])
            cell_candidate_profile_overlaps[cand] = cumulative_max_per_cell_overlap
            
        # Normalize within each cell
        denominator = sum(cell_candidate_profile_overlaps.values())
        for cand in cell_candidate_profile_overlaps:
            cell_candidate_profile_overlaps[cand] /= denominator
        cells_candidate_profile_overlaps.append(cell_candidate_profile_overlaps)
        
    return cells_candidate_profile_overlaps

def profile_intersect_size_w_gt_neighbors_candidate_scoring(cells, cells_gt, profile_dict, max_labels_in_profile=5):
    profiles = {cand : set(get_entity_profile(profile_dict, cand, max_labels_in_profile)) for candidates in cells for cand in candidates}
    cells_candidate_profile_overlaps = []
    for cur_cell_idx, candidates in enumerate(cells):
        cell_candidate_profile_overlaps = {}
        for cand in candidates:
            cand_profile = profiles[cand]
            
            if len(cand_profile) == 0:
                cell_candidate_profile_overlaps[cand] = 0
                continue
                
            cumulative_cell_overlap = 0
            for cell_idx in range(len(cells)):
                # Only compare to candidates in other cells
                if cell_idx == cur_cell_idx:
                    continue
                cumulative_cell_overlap += len(cand_profile & profiles[cells_gt[cell_idx]])
            cell_candidate_profile_overlaps[cand] = cumulative_cell_overlap
            
        # Normalize within each cell
        denominator = sum(cell_candidate_profile_overlaps.values())
        for cand in cell_candidate_profile_overlaps:
            cell_candidate_profile_overlaps[cand] /= denominator
        cells_candidate_profile_overlaps.append(cell_candidate_profile_overlaps)
        
    return cells_candidate_profile_overlaps

# Embedding methods
def random_amongst_embedding_ents(cells, embeddings):
    candidate_similarities = []
    for cell_idx, candidates in enumerate(cells):
        cell_cand_sims = {cand : 1 if cand in embeddings else 0 for cand in candidates}
        # normalize within each cell
        denominator = sum(cell_cand_sims.values())
        for cand in cell_cand_sims:
            cell_cand_sims[cand] /= denominator
        candidate_similarities.append(cell_cand_sims)
    
    return candidate_similarities

def avg_cell_embedding_sim_candidate_scoring(cells, embeddings):
    # We will essentially ignore entities for which we don't have embeddings
    cells_candidate_embeddings = [[list(embeddings[c]) for c in candidates if c in embeddings] for candidates in cells]
    
    candidate_similarities = []
    for cell_idx, candidates in enumerate(cells):
        other_cell_cand_embeds = [emb for i in range(len(cells)) if i != cell_idx for emb in cells_candidate_embeddings[i]]
        cell_cand_sims = {cand : np.sum(embeddings.cosine_similarities(embeddings[cand], other_cell_cand_embeds))
                          if cand in embeddings else 0 for cand in candidates}
        # normalize within each cell
        denominator = sum(cell_cand_sims.values())
        for cand in cell_cand_sims:
            cell_cand_sims[cand] /= denominator
        candidate_similarities.append(cell_cand_sims)
    
    return candidate_similarities

def max_cell_embedding_sim_candidate_scoring(cells, embeddings):
    # We will essentially ignore entities for which we don't have embeddings
    cells_candidate_embeddings = [[list(embeddings[c]) for c in candidates if c in embeddings] for candidates in cells]
    
    candidate_similarities = []
    for cur_cell_idx, candidates in enumerate(cells):
        cell_cand_sims = {}
        for cand in candidates:
            if cand not in embeddings:
                cell_cand_sims[cand] = 0
                continue
                
            cumulative_max_per_cell_sim = 0
            for cell_idx in range(len(cells)):
                # only compare to candidates in other cells
                if cell_idx == cur_cell_idx:
                    continue
                cumulative_max_per_cell_sim += max(embeddings.cosine_similarities(embeddings[cand], cells_candidate_embeddings[cell_idx]))
            cell_cand_sims[cand] = cumulative_max_per_cell_sim

        # normalize within each cell
        denominator = sum(cell_cand_sims.values())
        for cand in cell_cand_sims:
            cell_cand_sims[cand] /= denominator
        candidate_similarities.append(cell_cand_sims)
    
    return candidate_similarities

def embedding_sim_w_gt_neighbors_candidate_scoring(cells, cells_gt, embeddings):
    candidate_similarities = []
    for cur_cell_idx, candidates in enumerate(cells):
        cell_cand_sims = {}
        for cand in candidates:
            if cand not in embeddings:
                cell_cand_sims[cand] = 0
                continue
                
            cumulative_cell_sim = 0
            for cell_idx in range(len(cells)):
                # only compare to candidates in other cells
                if cell_idx == cur_cell_idx:
                    continue
                # only compare to gt neighbors that have an embedding in our embedding model
                if cells_gt[cell_idx] not in embeddings:
                    continue
                
                cumulative_cell_sim += embeddings.similarity(cand, cells_gt[cell_idx])
            cell_cand_sims[cand] = cumulative_cell_sim

        # normalize within each cell
        denominator = sum(cell_cand_sims.values())
        for cand in cell_cand_sims:
            cell_cand_sims[cand] /= denominator
        candidate_similarities.append(cell_cand_sims)
    
    return candidate_similarities


#=============================
# Result evaluation helpers
#=============================

def choose_best_candidates_from_scores(scores):
    choices = []
    for cell_candidate_scores in scores:
        # we need to break ties randomly since the input data may not be formatted randomly
        # This is specifically the case with the 'chiefs' data - first row of each cell is the GT
        max_score = max(cell_candidate_scores.values())
        choices.append(random.choice([candidate for candidate, score in cell_candidate_scores.items() if score == max_score]))
    return choices

def choose_gt_if_any_method_chooses_it(method_scores, cells_gt):
    choices = []
    for cell_idx in range(len(cells_gt)):
        cell_gt = cells_gt[cell_idx]
        choices.append("")
        for scores in method_scores:
            cell_cand_scores = scores[cell_idx]
            # we need to break ties randomly since the input data may not be formatted randomly
            # This is specifically the case with the 'chiefs' data - first row of each cell is the GT
            max_score = max(cell_cand_scores.values())
            cell_choice = random.choice([candidate for candidate, score in cell_cand_scores.items() if score == max_score])
            if cell_choice == cell_gt:
                choices[-1] = cell_choice
                break
    return choices

def choose_gt_if_all_methods_chooses_it(method_scores, cells_gt):
    choices = []
    for cell_idx in range(len(cells_gt)):
        cell_gt = cells_gt[cell_idx]
        choices.append(cell_gt)
        for scores in method_scores:
            cell_cand_scores = scores[cell_idx]
            # we need to break ties randomly since the input data may not be formatted randomly
            # This is specifically the case with the 'chiefs' data - first row of each cell is the GT
            max_score = max(cell_cand_scores.values())
            cell_choice = random.choice([candidate for candidate, score in cell_cand_scores.items() if score == max_score])
            if cell_choice != cell_gt:
                choices[-1] = ""
                break
    return choices

def get_margin_stats(scores, cells_gt):
    margins = []
    for cell_idx in range(len(cells_gt)):
        cell_scores = scores[cell_idx]
        best_incorrect_score = max([cell_scores[cand] for cand in cell_scores if cand != cells_gt[cell_idx]])
        gt_score = cell_scores[cells_gt[cell_idx]]
        margins.append(gt_score - best_incorrect_score)
        
    margin_stats = []
    margin_stats.append("{:.4f}".format(np.mean(margins)))
    margin_stats.append("{:.4f}".format(np.min(margins)))
    margin_stats.append("{:.4f}".format(np.percentile(margins,25)))
    margin_stats.append("{:.4f}".format(np.percentile(margins,50)))
    margin_stats.append("{:.4f}".format(np.percentile(margins,75)))
    margin_stats.append("{:.4f}".format(np.max(margins)))
    
    return margin_stats

def print_f1_and_margin_stats_for_method_scores(scores_by_method, num_trials=10):
    # headers = ["Method", "Precision", "Recall", "F1"]
    headers = ["Method", "F1", "Margin Avg", "Min", "25%", "50%", "75%", "Max"] # just looking at one metric for now since they are currently all the same
    rows = []
    num_trials = 10
    for method, scores in scores_by_method.items():
    #     avg_precision=0
    #     avg_recall=0
        avg_f1=0
        for i in range(num_trials):
            choices = choose_best_candidates_from_scores(scores)
    #         avg_precision += precision_score(choices, cells_gt, average="micro")
    #         avg_recall += recall_score(choices, cells_gt, average="micro")
            avg_f1 += f1_score(choices, cells_gt, average="micro")
    #     avg_precision /= num_trials
    #     avg_recall /= num_trials
        avg_f1 /= num_trials

        margin_stats = get_margin_stats(scores, cells_gt)

    #     rows.append([method,
    #                  "{:.2f}".format(avg_precision),
    #                  "{:.2f}".format(avg_recall),
    #                  "{:.2f}".format(avg_f1)
    #                 ])
        rows.append([method, "{:.2f}".format(avg_f1)] + margin_stats)

    # adjust table spacing for organization
    for idx in range(len(rows)-3,0,-3):
        rows.insert(idx,["",""])

    print(tabulate(rows, headers=headers))
    
#=======================
# Agreement Analysis
#=======================
def print_cell_agreement_of_methods(method_scores, num_trials=10):
    headers = ([""] + list(method_scores.keys()))
    rows = []

    for method_r in method_scores.keys():
        scores_r = method_scores[method_r]
        row = [method_r]
        for method_c in method_scores.keys():
            scores_c = method_scores[method_c]



            avg_agree_correct=0
            avg_agree_incorrect=0
            for i in range(num_trials):
                choices_r = choose_best_candidates_from_scores(scores_r)
                choices_c = choose_best_candidates_from_scores(scores_c)
                for j in range(len(cells_gt)):
                    if choices_r[j] != choices_c[j]:
                        continue
                    if choices_r[j] == cells_gt[j]:
                        avg_agree_correct += 1
                    else:
                        avg_agree_incorrect += 1
            avg_agree_correct /= num_trials
            avg_agree_incorrect /= num_trials
            row.append("{},\n{}".format(avg_agree_correct,avg_agree_incorrect))
        rows.append(row)

    print("# cells agreed upon: 'correct, incorrect'")
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    
def print_f1_of_method_combs(method_scores, combs, num_trials=10):
    headers = ["Methods", "F1 of logical OR", "F1 of logical AND"] # just looking at one metric for now since they are currently all the same
    rows = []
    for method_comb in combs:
        comb_scores = [method_scores[method] for method in method_comb]
        
        avg_OR_f1=0
        avg_AND_f1=0
        for i in range(num_trials):
            OR_choices = choose_gt_if_any_method_chooses_it(comb_scores, cells_gt)
            AND_choices = choose_gt_if_all_methods_chooses_it(comb_scores, cells_gt)
            avg_OR_f1 += f1_score(OR_choices, cells_gt, average="micro")
            avg_AND_f1 += f1_score(AND_choices, cells_gt, average="micro")
        avg_OR_f1 /= num_trials
        avg_AND_f1 /= num_trials
        
        comb_str = ", ".join(method_comb)
        rows.append([comb_str, "{:.2f}".format(avg_OR_f1), "{:.2f}".format(avg_AND_f1)])
    
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    