# Copyright (c) 2023-2025 The pymovements Project Authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Provides dataset definitions."""
from pymovements.datasets.bsc import BSC
from pymovements.datasets.bsc2 import BSCII
from pymovements.datasets.chinese_reading import ChineseReading
from pymovements.datasets.codecomprehension import CodeComprehension
from pymovements.datasets.colagaze import CoLAGaze
from pymovements.datasets.copco import CopCo
from pymovements.datasets.daemons import DAEMONS
from pymovements.datasets.didec import DIDEC
from pymovements.datasets.emtec import EMTeC
from pymovements.datasets.etdd70 import ETDD70
from pymovements.datasets.fakenews import FakeNewsPerception
from pymovements.datasets.gaze4hate import Gaze4Hate
from pymovements.datasets.gaze_graph import GazeGraph
from pymovements.datasets.gaze_on_faces import GazeOnFaces
from pymovements.datasets.gazebase import GazeBase
from pymovements.datasets.gazebasevr import GazeBaseVR
from pymovements.datasets.hbn import HBN
from pymovements.datasets.iitb_hgc import IITB_HGC
from pymovements.datasets.interead import InteRead
from pymovements.datasets.judo1000 import JuDo1000
from pymovements.datasets.mecol1w1 import MECOL1W1
from pymovements.datasets.mecol2w1 import MECOL2W1
from pymovements.datasets.mecol2w2 import MECOL2W2
from pymovements.datasets.mousecursor import MouseCursor
from pymovements.datasets.onestop import OneStop
from pymovements.datasets.potec import PoTeC
from pymovements.datasets.potsdam_binge_remote_pvt import PotsdamBingeRemotePVT
from pymovements.datasets.potsdam_binge_wearable_pvt import PotsdamBingeWearablePVT
from pymovements.datasets.provo import Provo
from pymovements.datasets.sb_sat import SBSAT
from pymovements.datasets.teco import TECO
from pymovements.datasets.toy_dataset import ToyDataset
from pymovements.datasets.toy_dataset_eyelink import ToyDatasetEyeLink
from pymovements.datasets.ucl import UCL


__all__ = [
    'BSC',
    'BSCII',
    'ChineseReading',
    'CodeComprehension',
    'CoLAGaze',
    'CopCo',
    'DAEMONS',
    'DIDEC',
    'EMTeC',
    'ETDD70',
    'FakeNewsPerception',
    'Gaze4Hate',
    'GazeBase',
    'GazeBaseVR',
    'GazeGraph',
    'GazeOnFaces',
    'HBN',
    'IITB_HGC',
    'InteRead',
    'JuDo1000',
    'MouseCursor',
    'MECOL1W1',
    'MECOL2W1',
    'MECOL2W2',
    'OneStop',
    'PoTeC',
    'PotsdamBingeRemotePVT',
    'PotsdamBingeWearablePVT',
    'Provo',
    'SBSAT',
    'TECO',
    'ToyDataset',
    'ToyDatasetEyeLink',
    'UCL',
]
