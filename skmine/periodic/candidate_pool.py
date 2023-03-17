import numpy as np

from .candidate import Candidate
from .class_patterns import mergeSortedPids, sortPids


class CandidatePool(object):
    """
    CandidatePool is a class for storing and managing Candidate objects.

    Attributes
    ----------
    candidates : dict
        A dictionary of Candidate objects stored using candidate IDs as keys.
    next_cid : int
        The ID to be used for the next created Candidate.
    cand_props : np.ndarray
        A np array containing the properties of all candidates. Each row represents a candidate and each column a
        different property. The first property is the "new" property, followed by the properties of the Candidate
        object.
    sorted_pids : list or range object
        A list or range object representing the IDs of all candidates, sorted in order of the "new" property and then
        in descending order of the "score" property.
    map_minorKs : dict
        A dictionary containing a mapping from minor keys to a dictionary that maps major keys to candidate IDs.
    map_nkeys : dict
        A dictionary that maps new keys to indices in the list_nkeys attribute.
    list_nkeys : list
        A list containing all new keys used by candidates.
    new_cids : dict
        A dictionary that maps new keys to a list of candidate IDs that are new according to that key.
    new_minorKs : dict
        A dictionary that maps new keys to a set of minor keys for candidates that are new according to that key.
    """

    def __init__(self, patts=[]):
        self.new_minorKs = None
        self.new_cids = None
        self.candidates = {}
        self.next_cid = 0
        self.cand_props = None
        self.sorted_pids = None
        self.map_minorKs = {}
        self.map_nkeys = {}
        self.list_nkeys = []
        self.resetNew()
        if len(patts) > 0:
            self.addCands(patts)

    def resetNew(self):
        """
        Resets the new_cids and new_minorKs attributes to empty dictionaries.
        """
        self.new_cids = {}
        self.new_minorKs = {}

    def getSortedPids(self):
        if self.sorted_pids is None:
            self.sorted_pids = sortPids(self.cand_props)
        return self.sorted_pids

    def getPidsForCid(self, cid):
        return np.where(self.cand_props[:, -1] == cid)[0]

    def getCidsForMinorK(self, mK):
        return list(self.map_minorKs.get(mK, {}).values())

    def getCandidate(self, cid):
        return self.candidates[cid]

    def getCandidates(self):
        return self.candidates

    def getNewKNum(self, nkey):
        return self.map_nkeys.get(nkey, -1)

    def getNewCids(self, nkey):
        return self.new_cids.get(nkey, [])

    def getNewMinorKeys(self, nkey):
        return self.new_minorKs.get(nkey, set())

    def nbNewCandidates(self, nkey=None):
        if nkey is None:
            return np.sum([len(v) for v in self.new_cids.values()])
        return len(self.new_cids.get(nkey, []))

    def getPropMat(self):
        return self.cand_props

    def getProp(self, pid, att=None):
        """
        Returns the property values for the specified candidate ID and property name or index.
        """
        if type(att) is int and att < len(Candidate.prop_list):
            return self.cand_props[pid, att]
        elif att in Candidate.prop_map:
            return self.cand_props[pid, Candidate.prop_map[att]]
        return self.cand_props[pid, :]

    def addCand(self, p, nkey=None, with_props=True):
        """
        Adds a new candidate to the candidate pool.

        Parameters
        ----------
        p : Candidate or tuple
            The candidate to add or a tuple to create a new candidate.
        nkey : any type
            The new key of the candidate.
        with_props : bool
            Whether to generate properties for the candidate.

        Returns
        -------
        Candidate or None
            The newly added candidate or None if candidate is a duplicate.
        """
        if type(p) is Candidate:
            c = p
        else:  # create candidate
            c = Candidate(-1, p)

        mK = c.getMinorKey()
        MK = c.getMajorKey()
        if mK in self.map_minorKs:
            if MK in self.map_minorKs[mK]:
                return None

        self.candidates[self.next_cid] = c
        c.setId(self.next_cid)

        # record event
        if mK not in self.map_minorKs:
            self.map_minorKs[mK] = {MK: self.next_cid}
        else:
            self.map_minorKs[mK][MK] = self.next_cid

        # record as new
        if nkey not in self.map_nkeys:
            self.map_nkeys[nkey] = len(self.list_nkeys)
            self.list_nkeys.append(nkey)
        if nkey not in self.new_minorKs:
            self.new_minorKs[nkey] = set()
        self.new_minorKs[nkey].add(mK)
        if nkey not in self.new_cids:
            self.new_cids[nkey] = []
        self.new_cids[nkey].append(self.next_cid)

        # generate properties
        if with_props:
            props = self.candidates[self.next_cid].getProps(
                self.map_nkeys[nkey])
            if self.cand_props is None:
                self.cand_props = np.array(props)
                self.sorted_pids = range(len(props))
            else:
                npids = range(
                    self.cand_props.shape[0], self.cand_props.shape[0] + len(props))
                self.cand_props = np.vstack([self.cand_props, props])
                self.sorted_pids = mergeSortedPids(
                    self.cand_props, self.sorted_pids, npids)

        self.next_cid += 1
        return c

    def addCands(self, ps, nkey=None, costOne=None):
        """
        Adds multiple candidates to the candidate pool.

        Parameters:
        ps (list): A list of candidate patterns.
        nkey (optional): The new key for the candidates, if any.
        costOne (optional): The cost of a substitution operation.
        """
        if nkey not in self.map_nkeys:
            self.map_nkeys[nkey] = len(self.list_nkeys)
            self.list_nkeys.append(nkey)
        props = []
        for p in ps:
            c = self.addCand(p, nkey, with_props=False)
            # HERE figure out reasonable max_offset
            max_offset = None
            if costOne is not None and c.getCost() is not None:
                max_offset = np.minimum(
                    int(np.floor(c.getCost() / costOne)), c.getNbOccs() - 2)
            if c is not None:
                props.extend(c.getProps(
                    self.map_nkeys[nkey], max_offset=max_offset))

        if len(props) == 0:
            return

        if self.cand_props is None:
            npids = range(len(props))
            self.cand_props = np.array(props)
            self.sorted_pids = sortPids(self.cand_props, npids)
        else:
            npids = range(
                self.cand_props.shape[0], self.cand_props.shape[0] + len(props))
            self.cand_props = np.vstack([self.cand_props, props])
            self.sorted_pids = mergeSortedPids(
                self.cand_props, self.sorted_pids, sortPids(self.cand_props, npids))