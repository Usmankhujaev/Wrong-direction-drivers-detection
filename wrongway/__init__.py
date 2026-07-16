"""Wrong-direction driver detection.

Implementation of "Real-Time, Deep Learning Based Wrong Direction Detection"
(Appl. Sci. 2020, 10, 2453), modernized: YOLO11 detection, ByteTrack/BoT-SORT
tracking, and the paper's entry-exit validation with zone, displacement, and
learned-flow direction models.
"""

__version__ = "2.0.0"
