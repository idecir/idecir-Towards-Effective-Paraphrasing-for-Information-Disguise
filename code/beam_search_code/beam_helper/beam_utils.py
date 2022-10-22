import math
from copy import deepcopy
from typing import Dict, List, Set, Tuple

import numpy as np

import beam_helper.fetch_use_scores as fetch_use_scores
import beam_helper.perplexity_calculation as perplexity_calculation

use_model = None
per_model = None
per_tokenizer = None
cf_model = None
bert_unmasker = None
faiss_model = None
bnp_model = None
checker_func = None


def reinit_models(a, b, c, d, e, f, g, h):
    # frugal way to prevent reloading of models when called via jupyter
    # notebook
    global use_model, per_model, per_tokenizer, cf_model, bert_unmasker, faiss_model, bnp_model, checker_func
    use_model = a
    per_model = b
    per_tokenizer = c
    cf_model = d
    bert_unmasker = e
    faiss_model = f
    bnp_model = g
    checker_func = h
    print("Models have been reinitialized")


class BeamSearch:
    """
    Given the input string, the BeamSearch Class runs a multi-phrase substitution based attack based on perplexitybased ranking of parse tree nodes AND BERT MLM and Counterfitting vector spaced synonyms based replacements.

    ** Parameters for Beam Search **:
    query_obj: Dictionary having attributes:
                "query_text": text of the `true sentence` ie the query
                "post_id": post ID of the source document in which the true sentence is present

    MAX_DEPTH: Number of levels in the beam search tree ie the MAXIMUM number of phrase substitutions allowed to be made in the query

    ALPHA_VAL:  Weighing parameter (to weight semantic similarity to the original query and locatibility differently).
                It is used in the calculation of score for a node in the BeamSearchTree.
                See the section `Algorithm Explanations` in the paper for the details.

    NUM_PERPLEXITY_NODES_TO_EXPAND: Number of nodes in the Constituency Parse Tree to be considered for attacking.
                                    Corresponds to the parameter "P"  in STEP 3 of Section 3.1  of the paper.

    BeamWidth: Max number of nodes at each level of the beam tree.

    NUM_FAISS_DOCS_TO_RETRIEVE: Max relevant documents to be fetched for the query in which the source document's presence needs to be checked.

    SIMILARITY_CUT_OFF_THRESHOLD:   Candidates which have a similarity of less than  `SIMILARITY_CUT_OFF_THRESHOLD` with the original sentence are filtered out.
                                    Corresponds to `epsilon` in the paper

    """

    def __init__(self,
                 query_obj: dict,
                 MAX_DEPTH: int,                          # 0 based indexing
                 # (1-alpha ) * sem + alpha * doc_score
                 ALPHA_VAL: float,
                 NUM_PERPLEXITY_NODES_TO_EXPAND: int = 4,
                 BEAM_WIDTH: int = 10,
                 NUM_FAISS_DOCS_TO_RETRIEVE: int = 20,
                 SIMILARITY_CUT_OFF_THRESHOLD: float = 0.8
                 ):
        '''Initialize all the variables.'''
        self.query_obj = deepcopy(query_obj)

        ###############################
        self._POST_ID: str = query_obj['post_id']
        self._MAX_DEPTH: int = MAX_DEPTH
        self._ALPHA_VAL: float = ALPHA_VAL
        self._NUM_PERPLEXITY_NODES_TO_EXPAND: int = NUM_PERPLEXITY_NODES_TO_EXPAND
        self._BEAM_WIDTH: int = BEAM_WIDTH
        self._NUM_FAISS_DOCS_TO_RETRIEVE: int = NUM_FAISS_DOCS_TO_RETRIEVE
        self._SIMILARITY_CUT_OFF_THRESHOLD: float = SIMILARITY_CUT_OFF_THRESHOLD

        ########################
        assert (MAX_DEPTH >= 1)
        assert (0 <= ALPHA_VAL and ALPHA_VAL <= 1)
        #######################

        print("All params have been initialized")

        self.original_sentence: str = query_obj['query_text']

        # Cache use embedding of original sentence
        self.use_embedding_org_sentence: np.ndarray = fetch_use_scores.fetch_use_embedding(
            use_model, self.original_sentence)

        # should contain details about each node selected in the beam at each
        # level
        self.beam_history = []

        # should contain a record of all adversarial sentences generated
        # (irrespective of whether it made to the next beam level for expansion
        # or not)
        self.contenders_history = [[]]

        # initialize root node at depth 0
        self._curr_beam_id = -1
        self.ROOT_BEAM_NODE = BeamNode(txt=self.original_sentence,
                                       level_id=0,
                                       beam_node_id=self.next_beam_id,
                                       parent_beam_node_id=-1,
                                       parent_leaf_contents=[],
                                       parent_restricted_leaf_ids=[],
                                       )

        self.curr_nodes_in_beam: List[BeamNode] = [self.ROOT_BEAM_NODE]

        # take the initial object (#assign variables where the most potent
        # attack is stored generally)
        self.best_via_sim = None        # If tie, on similarity, then order on basis of rank
        self.best_via_rank = None       # If tie, on rank, then order on basis of similarity
        self.best_via_hybrid = None     # If tie, on f_value, then order on basis of rank
        #######################################################################
        # stores the best attacks (ie values of `best_via_sim`,`best_via_rank`
        # and `best_via_hybrid``) for each depth of the beam search tree
        self.level_history = [[]]

    @property
    def next_beam_id(self) -> int:
        self._curr_beam_id += 1
        return self._curr_beam_id

    def remove_duplicates_from_beam_level(self):
        """Prevents duplicate candidates (generated via different paths) to be included in the next beam level
        """
        tmp_set = set()
        indices_to_have = []
        for idx, x in enumerate(self.beam_contenders_pool):
            if x['sentence'].lower() in tmp_set:
                continue
            tmp_set.add(x['sentence'].lower())
            indices_to_have.append(idx)

        self.beam_contenders_pool = list(
            map(lambda x: self.beam_contenders_pool[x], indices_to_have))

    def run_beam_search(self):

        org_sentence = ' '.join(self.ROOT_BEAM_NODE.bnp_node.leaf_contents)
        already_considered: Set[str] = set([org_sentence.lower()])

        for curr_depth_being_generated in range(1, self._MAX_DEPTH + 1):

            print(
                f"Expanding nodes of depth: {curr_depth_being_generated-1} to generate nodes for depth: {curr_depth_being_generated}",
                flush=True)
            # print("Number of nodes in current beam is: ", len(self.curr_nodes_in_beam))

            self.beam_contenders_pool: List[Dict] = []

            ###################################################################
            for curr_node in self.curr_nodes_in_beam:
                '''Create parse tree based on perplexity scores'''
                curr_node.make_perplexity_based_parse_tree()

                '''Decide what are the important parts of the sentence to expand based on the heuristic decided by you'''
                curr_node.choose_promising_nodes(
                    self._NUM_PERPLEXITY_NODES_TO_EXPAND)

                '''Expand the promising nodes ie derive their BERT suggestions, Counterfitting SUggestions. For these suggestions, also calculate their USE scores and their retrieval based scores'''
                curr_node.expand_promising_nodes()

                curr_node.rate_promising_nodes(
                    self.use_embedding_org_sentence, self._POST_ID)

                '''Append these promising nodes into the common pool'''
                for curr_attack_node in curr_node.bnp_node.expanded_nodes:
                    self.beam_contenders_pool.extend(
                        curr_attack_node['bert_suggestions'])
                    self.beam_contenders_pool.extend(
                        curr_attack_node['cf_suggestions'])
            ###################################################################

            '''Filter out the really bad candidates'''
            print(
                f"Num candidates in the pool for level {curr_depth_being_generated}: {len(self.beam_contenders_pool)}")

            # Done:  FIlter out beam candidates whose sentences have alrwady
            # been expanded as part of some other node
            self.beam_contenders_pool = list(filter(lambda x: x['use_score'] >= self._SIMILARITY_CUT_OFF_THRESHOLD,
                                                    self.beam_contenders_pool))

            self.beam_contenders_pool = list(filter(lambda x: x['sentence'].lower() not in already_considered,
                                                    self.beam_contenders_pool))
            print(
                f"[after filter] Num candidates in the pool for level {curr_depth_being_generated}: {len(self.beam_contenders_pool)}")

            # Calculate f value for each candidate
            list(map(lambda x: self.rate_candidate(x), self.beam_contenders_pool))
            # print("keys are: ", self.beam_contenders_pool[0].keys())

            self.remove_duplicates_from_beam_level()

            '''Update the best attacks'''
            self.update_best_attacks()

            self.level_history.append(
                {
                    "best_sim": self.best_via_sim,
                    "best_rank": self.best_via_rank,
                    "best_hybrid": self.best_via_hybrid})

            '''Rank the current promising nodes based on your ALPHA based heuristic'''
            # self.beam_contenders_pool = sorted(self.beam_contenders_pool, key=lambda x:-x['f_score'])
            self.contenders_history.append(self.beam_contenders_pool)

            # Policy to choose nodes which go the next level
            # self.beam_contenders_pool = self.beam_contenders_pool[:self._BEAM_WIDTH]
            self.limit_beam_level()

            for curr_beam_node in self.beam_contenders_pool:
                already_considered.add(curr_beam_node['sentence'].lower())

            '''Define the BEAM NODES for the next level'''
            self.beam_history.append([x.json_version()
                                     for x in self.curr_nodes_in_beam])

            self.curr_nodes_in_beam = [
                BeamNode(
                    txt=x['sentence'],
                    level_id=curr_depth_being_generated,
                    beam_node_id=self.next_beam_id,
                    parent_beam_node_id=x['par_beam_node'],
                    parent_leaf_contents=x['parent_leaf_contents'],
                    parent_restricted_leaf_ids=x['parent_restricted_leaf_ids'],
                    attack_node=x['attack_node'],
                    s_t=x['s_t']) for x in self.beam_contenders_pool]

            print(
                "###############  DEPTH DONE ###################################",
                flush=True)

        self.beam_history.append([x.json_version()
                                 for x in self.curr_nodes_in_beam])
        print("BEAM Search has run.")

    def limit_beam_level(self):
        """Heuristic choosing of selecting nodes for the next level of the beam search tree.
        """
        if len(self.beam_contenders_pool) <= self._BEAM_WIDTH:
            return

        self.beam_contenders_pool = [
            (idx, x) for idx, x in enumerate(
                self.beam_contenders_pool)]

        IDS_TO_INCLUDE = set()

        # 2 sim, 2 doc, 6 from everywhere else
        self.beam_contenders_pool = sorted(
            self.beam_contenders_pool,
            key=lambda x: -x[1]['use_score'])
        IDS_TO_INCLUDE.add(self.beam_contenders_pool[0][0])
        IDS_TO_INCLUDE.add(self.beam_contenders_pool[1][0])

        self.beam_contenders_pool = sorted(
            self.beam_contenders_pool,
            key=lambda x: -
            x[1]["q_scores"]["best_doc_rank"])
        IDS_TO_INCLUDE.add(self.beam_contenders_pool[0][0])
        IDS_TO_INCLUDE.add(self.beam_contenders_pool[1][0])

        self.beam_contenders_pool = sorted(
            self.beam_contenders_pool,
            key=lambda x: -x[1]["f_score"])
        for idx, elem in self.beam_contenders_pool:
            if len(IDS_TO_INCLUDE) >= self._BEAM_WIDTH:
                break
            if idx not in IDS_TO_INCLUDE:
                IDS_TO_INCLUDE.add(idx)

        # print("IDS to include is: ", IDS_TO_INCLUDE)
        self.beam_contenders_pool = list(
            filter(
                lambda x: x[0] in IDS_TO_INCLUDE,
                self.beam_contenders_pool))
        self.beam_contenders_pool = [x[1] for x in self.beam_contenders_pool]

    def rate_candidate(self, doc: Dict):
        doc["f_score"] = self.calculate_f_val(
            doc['use_score'], doc['q_scores']['best_doc_rank'])

    def calculate_f_val(self, semantic_sim_score: float,
                        best_doc_rank: int) -> float:
        """
        Args:
            semantic_sim_score (float): semantic similarity of the candidate perturbation with the original query
            best_doc_rank (int): The rank of the source document for the original query

        Returns:
            float: Weighted score for the candidate (Corresponds to f(s) in section 3.2 of the paper)
        """

        if best_doc_rank == 10**9:
            # this means that the source document was not PRESENT in the top
            # `NUM_FAISS_DOCS_TO_RETRIEVE`
            best_doc_rank = self._NUM_FAISS_DOCS_TO_RETRIEVE + 1

        # normalize the rank
        doc_score = math.fabs(best_doc_rank - 1) / \
            self._NUM_FAISS_DOCS_TO_RETRIEVE

        f_val = (1 - self._ALPHA_VAL) * (semantic_sim_score) + \
            self._ALPHA_VAL * doc_score
        return f_val

    def update_best_attacks(self):
        """
        Updates best candidates after every beam level
        """
        if len(self.beam_contenders_pool) == 0:
            return

        ###################
        sim_score, doc_score, sim_idx = max(
            [
                (x['use_score'], x['q_scores']['best_doc_rank'], idx) for idx, x in enumerate(
                    self.beam_contenders_pool)])
        if self.best_via_sim is None:
            self.best_via_sim = self.beam_contenders_pool[sim_idx]
        else:
            if (sim_score,
                doc_score) > (self.best_via_sim['use_score'],
                              self.best_via_sim['q_scores']['best_doc_rank']):
                self.best_via_sim = self.beam_contenders_pool[sim_idx]
        del sim_idx
        ##################
        ###################
        doc_score, sim_score, rank_idx = max(
            [
                (x['q_scores']['best_doc_rank'], x['use_score'], idx) for idx, x in enumerate(
                    self.beam_contenders_pool)])
        if self.best_via_rank is None:
            self.best_via_rank = self.beam_contenders_pool[rank_idx]
        else:
            if (doc_score,
                sim_score) > (self.best_via_sim['q_scores']['best_doc_rank'],
                              self.best_via_rank['use_score']):
                self.best_via_rank = self.beam_contenders_pool[rank_idx]
        del rank_idx
        ##################
        ###################
        f_score, doc_score, f_idx = max(
            [
                (x['f_score'], x['q_scores']['best_doc_rank'], idx) for idx, x in enumerate(
                    self.beam_contenders_pool)])
        if self.best_via_hybrid is None:
            self.best_via_hybrid = self.beam_contenders_pool[f_idx]
        else:
            if (f_score,
                doc_score) > (self.best_via_hybrid['f_score'],
                              self.best_via_hybrid['q_scores']['best_doc_rank']):
                self.best_via_hybrid = self.beam_contenders_pool[f_idx]
        del f_idx
        ##################

    def fetch_consolidated_results(self):
        ans = self.query_obj
        ans["beam_history"] = self.beam_history
        ans['contenders_history'] = self.contenders_history

        for level in ans['contenders_history']:
            for obj in level:
                del obj['parent_leaf_contents']
                del obj['parent_restricted_leaf_ids']
        ans['level_history'] = self.level_history
        return ans


class BeamNode:
    """Represents a node in the Beam Search Tree.
    """

    def __init__(self,
                 txt: str,                                # text which needs to be expanded and parsed
                 level_id: int,                           # what level of perturbation is this
                 beam_node_id: int,                       # node iD within the entire beam tree
                 parent_beam_node_id: int,                # node ID of parent
                 # contains the content of the leaf of the parent
                 parent_leaf_contents: List[str],
                 # contains the tokens which have already been perturbed and
                 # won't be allowed to be perturbed in subsequent depths
                 parent_restricted_leaf_ids: List[int],
                 # node ID in the Berkley Neural Parser Tree of the parent from
                 # which this new node has arisen (can be same for multiple
                 # nodes within the same beam)
                 attack_node: int = -1,
                 # whether this node was generated as a result of BERT, CF
                 # perturbation (NA in case of root ie original sentence)
                 s_t="NA"
                 ):

        self.text: str = txt
        # level of the beam search tree at which this node is present
        self.level_id: int = level_id
        # node ID assigned to this node in the beam search tree
        self.beam_node_id: int = beam_node_id
        # node ID assigned to the parent in the beam search tree
        self.parent_beam_id: int = parent_beam_node_id
        self.attack_node: int = attack_node
        self.s_t: str = s_t

        '''create the Berkley Neural Parser tree for this Node'''
        doc = bnp_model(self.text)
        sent = list(doc.sents)[0]
        self.bnp_node = BnpNode(
            sent,
            is_root=True,
            beam_node_id=self.beam_node_id,
            parent_leaf_contents=parent_leaf_contents,
            parent_restricted_leaf_ids=parent_restricted_leaf_ids)

    def make_perplexity_based_parse_tree(self):
        # print(f"Building perplexity based parse tree for: {self.beam_node_id}")
        self.bnp_node.build_parse_tree()

    def choose_promising_nodes(self, NUM_PERPLEXITY_NODES_TO_EXPAND: int):
        # print(f"Shortlisting areas to attack for: {self.beam_node_id}")
        self.bnp_node.decide_chosen_nodes(NUM_PERPLEXITY_NODES_TO_EXPAND)

    def expand_promising_nodes(self):
        # print(f"Expanding chosen attack nodes for: {self.beam_node_id}")
        self.bnp_node.expand_chosen_nodes()

    def rate_promising_nodes(
            self, use_embedding_of_original_sentence: np.ndarray,
            POST_ID: str):
        # go through all the expanded nodes
        for curr_node in self.bnp_node.expanded_nodes:
            for suggestion_type in ['bert_suggestions', 'cf_suggestions']:
                S_TYPE = suggestion_type.split("_")[0]
                for curr_obj in curr_node[suggestion_type]:
                    # TODO: Replace constant
                    curr_obj['par_beam_node'] = self.beam_node_id
                    curr_obj['attack_node'] = curr_node["node_id"]
                    curr_obj['s_t'] = S_TYPE
                    curr_obj['masked_sen'] = curr_node["masked_sentence"]
                    curr_obj['parent_leaf_contents'] = self.bnp_node.leaf_contents
                    curr_obj['parent_restricted_leaf_ids'] = self.bnp_node.restricted_leaf_ids
                    curr_obj["q_scores"] = faiss_model.fetch_results(
                        curr_obj['sentence'], POST_ID, 20)
                    use_embedding = fetch_use_scores.fetch_use_embedding(
                        use_model, curr_obj['sentence'])

                    curr_obj["use_score"]: float = float(
                        np.inner(use_embedding, use_embedding_of_original_sentence))

    def json_version(self):
        obj = {}
        obj["text"]: str = self.text
        obj["level_id"]: int = self.level_id
        obj["beam_node_id"]: int = self.beam_node_id
        obj["parent_beam_id"]: int = self.parent_beam_id
        obj["attack_node"]: int = self.attack_node
        obj["s_t"]: int = self.s_t

        if hasattr(self.bnp_node, "parse_tree"):
            obj['parse_tree'] = self.bnp_node.parse_tree

        ###################################
        # seeing if allowed to change
        # obj['permissions'] = [[x, False] for x in self.bnp_node.leaf_contents]
        # for curr_elem in self.bnp_node.restricted_leaf_ids:
        #     print("Curr elem is ", curr_elem)
        #     obj['permissions'][curr_elem][1]=True
        ############################

        return obj


def find_lcp(arr1: List[str], arr2: List[str]) -> int:
    '''Returns number of tokens in longest common prefix'''
    ans = -1
    for i in range(min(len(arr1), len(arr2))):
        if arr1[i] == arr2[i]:
            ans = i
        else:
            break
    ans += 1
    return ans


def fetch_updated_restricted_ids(
        parent_leaf_contents: List[str],
        own_leaf_contents: List[str],
        parent_restrictions: List[int]) -> List[int]:
    """fetches the restrictions regarding which nodes in the parse tree are not eligible for attack as an attack had already been made on them in the past.

    Args:
        parent_leaf_contents (List[str]): Contents of the leafs of the parse tree of the parent
        own_leaf_contents (List[str]): Contents of the leafs of the parse tree of the current node
        parent_restrictions (List[int]): Leaf IDs of the parent which have already been modified

    Returns:
        List[int]: Leaf IDs in the current node which should not be modifiable via an attack
    """
    if parent_leaf_contents == []:
        print("PARENT LEAF CONTENTS IS EMPTY")
        return []

    # print("###############################")
    # print("Parent state:")
    # print(*list(zip(parent_leaf_contents,[True if i in parent_restrictions else False for i in range(len(parent_leaf_contents))] )), sep="\n")

    own_restrictions: List[int] = []
    # findest prefix ub
    num_tokens_par = len(parent_leaf_contents)
    num_tokens_self = len(own_leaf_contents)

    prefix_common_tokens = find_lcp(parent_leaf_contents, own_leaf_contents)

    parent_leaf_contents.reverse()
    own_leaf_contents.reverse()
    suffix_common_tokens = find_lcp(parent_leaf_contents, own_leaf_contents)

    parent_leaf_contents.reverse()
    own_leaf_contents.reverse()

    # print("Common tokens at prefix and suffix is: ",prefix_common_tokens, suffix_common_tokens)

    # find offset
    replaced_size = num_tokens_par - \
        (prefix_common_tokens + suffix_common_tokens)
    replacement_size = num_tokens_self - \
        (prefix_common_tokens + suffix_common_tokens)

    offset = replacement_size - replaced_size

    idx_from_which_offset_to_add = (
        num_tokens_par - 1) - (suffix_common_tokens) + 1

    for curr_num in parent_restrictions:
        # add prefix based restrictions
        if curr_num < prefix_common_tokens:
            own_restrictions.append(curr_num)
        elif curr_num >= idx_from_which_offset_to_add:
            # add suffix based restrictions
            own_restrictions.append(curr_num + offset)

    # add new restrictions
    starting_token = prefix_common_tokens
    ending_token = (num_tokens_self - 1) - (suffix_common_tokens) + 1 - 1

    for i in range(starting_token, ending_token + 1):
        own_restrictions.append(i)
        # print("Preventing change of : ", own_leaf_contents[i])

    # print("Child state:")
    # print(*list(zip(own_leaf_contents,[True if i in own_restrictions else False for i in range(len(own_leaf_contents))] )), sep="\n")
    # print("###############################")

    return own_restrictions
    # find suffix lb


class BnpNode:
    """Represents a node in the parse tree for a sentence (as obtained by the Berkeley Neural Parser).
    """

    def __init__(self,
                 # Spacy based span which contains details about the parse tree
                 span_obj,
                 # there are several nodes in the Berkley Parse tree. Only the
                 # origin of the parse tree has this attribute set as `True`
                 is_root: bool = False,
                 # The BEAM NODE ID in the beam tree to which this parse tree
                 # belongs. This is `-1` only for thr root of the parse tree
                 beam_node_id: int = -1,
                 # The BEAM NODE ID in the beam tree to which this parse tree
                 # belongs. This is `-1` only for thr root of the parse tree
                 parent_leaf_contents: List[str] = [],
                 parent_restricted_leaf_ids: List[int] = []):              # the token IDs in the `parent_leaf_contents` which need to be IGNORED (should not be perturbed)

        self.span_obj = span_obj
        self.parsed_string: str = str(self.span_obj)

        self.span_children = list(span_obj._.children)
        self.node_children: List[BnpNode] = []

        self.num_children: int = len(list(self.span_obj._.children))

        # node ID within the parse tree (is correctly set in `int_dfs`)
        self.node_id = -1
        # leaf ID (if applicable) within the parse tree (is correctly set in
        # `int_dfs`)
        self.leaf_id: int = -1
        self.min_leaf_id: int = 10**9
        self.max_leaf_id: int = -1

        self._is_root: bool = is_root

        # (NODE ID, Potential for counterfitting)
        if self._is_root:
            self.ranked_nodes: List[Tuple[int, bool, float]] = []
            self.chosen_nodes: List[Dict] = []
            self.beam_node_id = beam_node_id

        send_dict = {"curr_node_id": 0, "curr_leaf_id": 0}

        if self._is_root:
            self.leaf_contents: List[str] = []
            self.init_dfs(send_dict, self.leaf_contents)
            self.parent_leaf_contents = parent_leaf_contents
            self.parent_restricted_leaf_ids = parent_restricted_leaf_ids

            # leaf contents have been made
            # TODO: Deal with restrictions
            self.restricted_leaf_ids = fetch_updated_restricted_ids(
                parent_leaf_contents, self.leaf_contents, self.parent_restricted_leaf_ids)

    @property
    def is_leaf(self):
        return True if self.num_children == 0 else False

    @property
    def curr_label(self):
        if self.is_leaf:
            all_tokens = self.span_obj._.parse_string.split("(")
            label = all_tokens[1].split(" ")[0]
        else:
            label = self.span_obj._.labels[0]
        return label

    def init_dfs(self, init_args, leaf_contents: List[str]):

        # init_args: dict of node id and leaf ids
        # returns (first leaf id. last leaf id)
        self.node_id = init_args['curr_node_id']
        init_args["curr_node_id"] += 1
        # print("NODE ID: ", self.node_id)
        if self.num_children != 0:

            # Not a leaf node, perform DFS further
            for curr_child in self.span_obj._.children:
                child_node = BnpNode(curr_child)
                min_leaf_in_child, max_leaf_in_child = child_node.init_dfs(
                    init_args, leaf_contents)
                self.min_leaf_id = min(self.min_leaf_id, min_leaf_in_child)
                self.max_leaf_id = max(self.max_leaf_id, max_leaf_in_child)

                self.node_children.append(child_node)
        else:
            # a leaf node, NO need to perform DFS any further
            self.leaf_id = init_args["curr_leaf_id"]
            leaf_contents.append(self.parsed_string)
            init_args["curr_leaf_id"] += 1
            self.min_leaf_id = self.leaf_id
            self.max_leaf_id = self.leaf_id
            # print("Parsed string at leaf is ", self.parsed_string)

        # print(f"Node: {self.node_id} : Sentence: {self.parsed_string}")
        return self.min_leaf_id, self.max_leaf_id

    def fetch_string(self, wanted_lb, wanted_ub) -> str:
        inter_lb, inter_ub = max(
            self.min_leaf_id, wanted_lb), min(
            self.max_leaf_id, wanted_ub)
        if inter_lb == self.min_leaf_id and inter_ub == self.max_leaf_id:
            return self.parsed_string
        if inter_ub < inter_lb:
            return ""

        result = ""
        for curr_child in self.node_children:
            result += " " + curr_child.fetch_string(inter_lb, inter_ub)

        if self.node_id == 0:
            result = ' '.join(result.split())
            result = result.lstrip()
            result = result.rstrip()
        return result

    def json_bert(self, ref_to_root):
        '''
        Responsible for constructing the sentences at each node, calculating perplexity for judging whether to expand the node or NOT
        '''

        obj = dict()
        obj["parsed_string"] = self.parsed_string
        obj["label"] = self.curr_label
        obj["num_children"] = len(list(self.span_obj._.children))
        obj["node_id"] = self.node_id
        obj["leaf_id"] = self.leaf_id
        obj["min_leaf_id"] = self.min_leaf_id
        obj["max_leaf_id"] = self.max_leaf_id

        obj["masked_sentence"] = ""

        '''If the node is worthy of expansion, then: do the following
        1) Calculate the masked version at this node
        2) Fill with BERT's top suggestion and calculate the perplexity score for this node
        '''

        if checker_func(self.curr_label):

            skip_due_to_repeatition = False
            for curr_elem in range(obj["min_leaf_id"], obj["max_leaf_id"] + 1):
                if curr_elem in ref_to_root.restricted_leaf_ids:
                    skip_due_to_repeatition = True
                    break
            if (not skip_due_to_repeatition):

                # fetch mask
                prefix_str = ""
                suffix_str = ""

                num_tokens = ref_to_root.max_leaf_id + 1
                prefix_ub = self.min_leaf_id - 1
                suffix_lb = self.max_leaf_id + 1

                if prefix_ub >= 0:
                    prefix_str = ref_to_root.fetch_string(0, prefix_ub)
                if suffix_lb < num_tokens:
                    suffix_str = ref_to_root.fetch_string(
                        suffix_lb, num_tokens - 1)

                # Storing prefix and suffix
                obj['prefix'] = prefix_str
                obj['suffix'] = suffix_str

                ###############################
                # print("parsed string is: ", obj["parsed_string"])
                # print("PREFIX is: ", obj["prefix"])
                # print("SUFFIX is: ", obj["suffix"])
                ##############################

                ans = prefix_str + " [MASK] " + suffix_str
                obj["masked_sentence"] = ' '.join(
                    ans.split())  # remove redundant pspaces

                # bert_suggestions = small_bert_unmasker(obj["masked_sentence"])
                # print("sentence is ", obj["masked_sentence"])
                bert_suggestions = bert_unmasker(obj["masked_sentence"])

                obj['first_bert_token'] = bert_suggestions[0]['token_str']
                obj['first_bert_sentence'] = obj["masked_sentence"].replace(
                    "[MASK]", obj['first_bert_token'])
                obj['node_perplexity']: float = perplexity_calculation.fetch_pll_scores(
                    per_tokenizer, per_model, obj['first_bert_sentence'])

                # does this node have potential for counterfitting based expansion
                # print("keys are: ", obj.keys())
                obj['cf_potential'] = obj["parsed_string"].lower(
                ) in cf_model.syn_dict

                ref_to_root.ranked_nodes.append(
                    (obj["node_id"], obj['cf_potential'], obj['node_perplexity']['final_score']))

        #########################################################
        # Repeat on CHILDREN
        obj["node_children"] = []
        for curr_child in self.node_children:
            obj["node_children"].append(curr_child.json_bert(ref_to_root))

        return obj

    def build_parse_tree(self):
        '''
        Builds a parse tree for the text. This tree will have various nodes representing different parts of the sentence that can be attacked. The perplexity scores for each valid node are also calculated.
        NOTE: Valid parts of the sentence to be attacked is determined by the `checker_func`
        '''
        self.parse_tree = self.json_bert(self)

    def decide_chosen_nodes(self, NUM_PERPLEXITY_NODES_TO_EXPAND: int):
        '''
        Rank Nodes on the basis of perplexity and choose the top `NUM_PERPLEXITY_NODES_TO_EXPAND` nodes for attacking (under constraints that atleast some are available for CF based attacks)
        '''
        print("Number of nodes to rank is: ", len(self.ranked_nodes))
        self.ranked_nodes = sorted(self.ranked_nodes, key=lambda x: -x[2])

        # Keep atleast one counterfitting
        self.chosen_nodes = self.ranked_nodes[:NUM_PERPLEXITY_NODES_TO_EXPAND]

        num_cf_based = 0

        for i in self.chosen_nodes:
            if i[1] == True:
                num_cf_based += 1

        if num_cf_based == 0:

            best_cf_node = None
            for i in self.chosen_nodes:
                if i[1] == True:
                    best_cf_node = i
                    break
            if best_cf_node is not None:
                print("Divine CF intervention done")
                self.chosen_nodes.pop()
                self.chosen_nodes.append(best_cf_node)

        self.chosen_nodes = set([x[0] for x in self.chosen_nodes])

    def expand_chosen_nodes(self):
        '''
        Expand the chosen top nodes. Expansion here means:
        1) Find their suggestions based on BERT masking model
        2) Find their suggestions based on CF model
        Stores the results in `self.expanded_nodes`

        '''
        # fetch JSON bert
        self.expanded_nodes: List[Dict] = []
        # NOTE: The nodes in `self.expanded_nodes` are not in increasing order
        # of preference
        self.dfs_to_expand_deserving_nodes(self.parse_tree)

    def dfs_to_expand_deserving_nodes(self, curr_node):

        if curr_node["node_id"] in self.chosen_nodes:
            # if deserving, add BnpNodeId to the
            new_obj = dict()
            new_obj['beam_node_id'] = self.beam_node_id

            # copy everything except children
            for key in curr_node:
                if key != "node_children":
                    new_obj[key] = curr_node[key]

            # expand the node with bert and cf suggestions
            new_obj['bert_suggestions'] = []
            new_obj['cf_suggestions'] = []

            # expand using bert (token, sentence)
            bert_suggestions = bert_unmasker(curr_node['masked_sentence'])
            new_obj['bert_suggestions'] = [
                {
                    "token": x['token_str'],
                    "sentence": curr_node['masked_sentence'].replace(
                        "[MASK]",
                        x['token_str'])} for x in bert_suggestions]
            del bert_suggestions

            # expand using counterfitting (token, sentence)
            if curr_node['cf_potential'] == True:
                # print("CF POTENTIAL DISCOVERED")
                top_words = cf_model.fetch_k_nearest(
                    curr_node['parsed_string'].lower(), 10)
                new_obj['cf_suggestions'] = [
                    {"token": w, "sentence": curr_node['masked_sentence'].replace("[MASK]", w)} for w in top_words]

            # add to results
            self.expanded_nodes.append(new_obj)

        # traverse through children
        for curr_child in curr_node["node_children"]:
            self.dfs_to_expand_deserving_nodes(curr_child)
