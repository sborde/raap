from signalizer.model.experiment.segments import SegmentMatrix
from signalizer.model.experiment.sequences import EventVector

loaders = {
    'event_vector': EventVector.load,
    'segment_matrix': SegmentMatrix.load,
}
