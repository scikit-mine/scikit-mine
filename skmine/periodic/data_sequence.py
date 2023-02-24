import numpy as np

from .class_patterns import makeOccsAndFreqs, cost_one


class DataSequence(object):
    """
    A class that constructs a data sequence from a list or a dictionary of events and their respective timestamps.

    Parameters
    ----------
    seq : list or dict
        The input sequence of events and timestamps.
        If `seq` is a list, it should contain tuples of (t, ev) where `t` is the timestamp and `ev` is the event name.
        If `seq` is a dictionary, it should have the event names as keys and the corresponding values should be a list
        of timestamps.

    Attributes
    ----------
    evStarts : dict
        A dictionary with the starting timestamps for each event.
    evEnds : dict
        A dictionary with the ending timestamps for each event.
    data_details : dict
        A dictionary containing information about the data, including the starting timestamp, ending timestamp,
        delta time, number of occurrences, original frequencies, adjusted frequencies and block delimiter.
    seqd : dict
        A dictionary where the keys are event numbers (integers) and the values are lists of timestamps.
    seql: list
        A list of tuples where the first element is the timestamp and the second element is the event number.
    list_ev : list
        A sorted list of unique event names.
    map_ev_num : dict
        A dictionary where the keys are event names and the values are event numbers.

    Examples
    --------
    Creating a `DataSequence` object with a list:

    >>> data = [(1, 'A'), (2, 'B'), (3, 'A'), (4, 'C')]
    >>> ds = DataSequence(data)
    """

    def __init__(self, seq):
        evNbOccs, evStarts, evEnds = ({}, {}, {})
        self.seqd = {}
        self.seql = []
        self.map_ev_num = {}
        self.list_ev = []

        seq_tmp = seq
        if type(seq) is list:
            seq_tmp = {}
            for (t, ev) in seq:
                if ev not in seq_tmp:
                    seq_tmp[ev] = []
                seq_tmp[ev].append(t)

        # construct list and dict, translating ev to num
        self.list_ev = sorted(seq_tmp.keys())
        self.map_ev_num = dict([(v, k) for (k, v) in enumerate(self.list_ev)])
        for q, dt in seq_tmp.items():
            self.seqd[self.map_ev_num[q]] = dt
        for (ev, ts) in self.seqd.items():
            self.seql.extend([(t, ev) for t in ts])
        self.seql.sort()

        for ev, sq in self.seqd.items():
            evNbOccs[ev] = len(sq)
            evStarts[ev] = np.min(sq)
            evEnds[ev] = np.max(sq)

        nbOccs, orgFreqs, adjFreqs, blck_delim = makeOccsAndFreqs(evNbOccs)
        t_end, t_start = (0, 0)
        if sum(evNbOccs.values()) > 0:
            t_end = np.max(list(evEnds.values()))
            t_start = np.min(list(evStarts.values()))
        deltaT = t_end - t_start
        self.evStarts = evStarts
        self.evEnds = evEnds
        self.data_details = {"t_start": t_start, "t_end": t_end, "deltaT": deltaT,
                             "nbOccs": nbOccs, "orgFreqs": orgFreqs, "adjFreqs": adjFreqs, "blck_delim": blck_delim}

    def getInfoStr(self):
        """
            Get information about the DataSequence
            |A|: number of events
            |O|: number of occurrences
            dT: delta time
            ({start_time} to {end_time})

            and then a description for each event
            {event_name} [{event_id}] (|O|={number_of_occurences} f={frequence} dt={delta_time}) ...
        """
        if self.data_details["nbOccs"][-1] == 0:
            ss = "-- Empty Data Sequence"
        else:
            ss = "-- Data Sequence |A|=%d |O|=%d dT=%d (%d to %d)" % (len(
                self.data_details["nbOccs"]) - 1, self.data_details["nbOccs"][-1], self.data_details["deltaT"],
                                                                      self.data_details["t_start"],
                                                                      self.data_details["t_end"])
            ss += "\n\t" + "\n\t".join(["%s [%d] (|O|=%d f=%.3f dT=%d)" % (
                self.list_ev[k], k, self.data_details["nbOccs"][k], self.data_details["orgFreqs"]
                [k], self.evEnds[k] - self.evStarts[k]) for k in
                                        sorted(range(len(self.list_ev)), key=lambda x: self.data_details["nbOccs"][x])])
        return ss

    def getEvents(self):
        return self.list_ev

    def getNumToEv(self):
        return dict(enumerate(self.list_ev))

    def getEvToNum(self):
        return self.map_ev_num

    def getSequenceStr(self, sep_te=" ", sep_o="\n", ev=None):
        if ev is None:
            return sep_o.join([("%s" + sep_te + "%s") % (t, self.list_ev[e]) for (t, e) in sorted(self.seql)])
        else:
            if ev in self.map_ev_num:
                return sep_o.join(["%s" % p for p in sorted(self.seqd.get(self.map_ev_num[ev], []))])
            else:
                return sep_o.join(["%s" % p for p in sorted(self.seqd.get(ev, []))])

    def getSequence(self, ev=None):
        if ev is None:
            return self.seql
        else:
            return self.seqd.get(ev, [])

    def getTend(self):
        return self.data_details["t_end"]

    def getDetails(self):
        return self.data_details

    def codeLengthResiduals(self):
        cl = 0
        for ev, ts in self.seqd.items():
            cl += len(ts) * cost_one(self.data_details, ev)
        return cl
