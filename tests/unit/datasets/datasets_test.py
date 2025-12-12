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
"""Test public dataset definitions."""
from __future__ import annotations

import pytest

import pymovements as pm
from pymovements import DatasetLibrary


@pytest.mark.parametrize(
    ('definition', 'dataset_name'),
    # please add datasets in alphabetical order
    [
        pytest.param(pm.datasets.BSC, 'BSC', id='BSC'),
        pytest.param(pm.datasets.BSCII, 'BSCII', id='BSCII'),
        pytest.param(pm.datasets.ChineseReading, 'ChineseReading', id='ChineseReading'),
        pytest.param(pm.datasets.CodeComprehension, 'CodeComprehension', id='CodeComprehension'),
        pytest.param(pm.datasets.CoLAGaze, 'CoLAGaze', id='CoLAGaze'),
        pytest.param(pm.datasets.CopCo, 'CopCo', id='CopCo'),
        pytest.param(pm.datasets.DAEMONS, 'DAEMONS', id='DAEMONS'),
        pytest.param(pm.datasets.DIDEC, 'DIDEC', id='DIDEC'),
        pytest.param(pm.datasets.EMTeC, 'EMTeC', id='EMTeC'),
        pytest.param(pm.datasets.ETDD70, 'ETDD70', id='ETDD70'),
        pytest.param(pm.datasets.FakeNewsPerception, 'FakeNewsPerception', id='FakeNewsPerception'),
        pytest.param(pm.datasets.Gaze4Hate, 'Gaze4Hate', id='Gaze4Hate'),
        pytest.param(pm.datasets.GazeBase, 'GazeBase', id='GazeBase'),
        pytest.param(pm.datasets.GazeBaseVR, 'GazeBaseVR', id='GazeBaseVR'),
        pytest.param(pm.datasets.GazeOnFaces, 'GazeOnFaces', id='GazeOnFaces'),
        pytest.param(pm.datasets.HBN, 'HBN', id='HBN'),
        pytest.param(pm.datasets.InteRead, 'InteRead', id='InteRead'),
        pytest.param(pm.datasets.IITB_HGC, 'IITB_HGC', id='IITB_HGC'),
        pytest.param(pm.datasets.JuDo1000, 'JuDo1000', id='JuDo1000'),
        pytest.param(pm.datasets.MECOL1W1, 'MECOL1W1', id='MECOL1W1'),
        pytest.param(pm.datasets.MECOL2W1, 'MECOL2W1', id='MECOL2W1'),
        pytest.param(pm.datasets.MECOL2W2, 'MECOL2W2', id='MECOL2W2'),
        pytest.param(pm.datasets.MouseCursor, 'MouseCursor', id='MouseCursor'),
        pytest.param(pm.datasets.OneStop, 'OneStop', id='OneStop'),
        pytest.param(pm.datasets.PoTeC, 'PoTeC', id='PoTeC'),
        pytest.param(
            pm.datasets.PotsdamBingeRemotePVT,
            'PotsdamBingeRemotePVT',
            id='PotsdamBingeRemotePVT',
        ),
        pytest.param(
            pm.datasets.PotsdamBingeWearablePVT,
            'PotsdamBingeWearablePVT',
            id='PotsdamBingeWearablePVT',
        ),
        pytest.param(pm.datasets.Provo, 'Provo', id='Provo'),
        pytest.param(pm.datasets.RaCCooNS, 'RaCCooNS', id='RaCCooNS'),
        pytest.param(pm.datasets.SBSAT, 'SBSAT', id='SBSAT'),
        pytest.param(pm.datasets.TECO, 'TECO', id='TECO'),
        pytest.param(pm.datasets.ToyDataset, 'ToyDataset', id='ToyDataset'),
        pytest.param(pm.datasets.ToyDatasetEyeLink, 'ToyDatasetEyeLink', id='ToyDatasetEyeLink'),
        pytest.param(pm.datasets.UCL, 'UCL', id='UCL'),
    ],
)
def test_public_dataset_registered(definition, dataset_name):
    assert dataset_name in DatasetLibrary.names()
    definition_from_library = DatasetLibrary.get(dataset_name)

    # simple equal between objects does not work as classes have different names.
    assert definition().to_dict() == definition_from_library.to_dict()
