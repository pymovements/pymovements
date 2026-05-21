# Copyright (c) 2023-2026 The pymovements Project Authors
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
"""Test for Screen class."""
import pytest

import pymovements as pm


def test_screen_init_without_attributes():
    screen = pm.Screen()
    assert isinstance(screen, pm.Screen)


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {},
            id='default',
        ),
        pytest.param(
            {'width_px': None, 'height_px': None},
            id='width-height-None',
        ),
        pytest.param(
            {'resolution': None},
            id='resolution-None',
        ),
        pytest.param(
            {'resolution': (None, None)},
            id='resolution-None-tuple',
        ),
    ],
)
def test_screen_init_resolution_is_none(kwargs):
    screen = pm.Screen(**kwargs)
    assert screen.resolution is None
    assert screen.width_px is None
    assert screen.height_px is None


@pytest.mark.parametrize(
    ('kwargs', 'expected_resolution'),
    [
        pytest.param(
            {'width_px': 1024},
            (1024, None),
            id='width',
        ),
        pytest.param(
            {'height_px': 768},
            (None, 768),
            id='height',
        ),
        pytest.param(
            {'width_px': 1024, 'height_px': 768},
            (1024, 768),
            id='width-height-1024x768',
        ),
        pytest.param(
            {'width_px': 1280, 'height_px': 1024},
            (1280, 1024),
            id='width-height-1280x1024',
        ),
        pytest.param(
            {'resolution': (1024, 768)},
            (1024, 768),
            id='resolution-1024x768',
        ),
        pytest.param(
            {'resolution': (1280, 1024)},
            (1280, 1024),
            id='resolution-1280x1024',
        ),
        pytest.param(
            {'resolution': [1024, 768]},
            (1024, 768),
            id='resolution-list',
        ),
    ],
)
def test_screen_init_has_correct_resolution(kwargs, expected_resolution):
    screen = pm.Screen(**kwargs)
    assert screen.resolution == expected_resolution
    assert screen.width_px == expected_resolution[0]
    assert screen.height_px == expected_resolution[1]


@pytest.mark.parametrize(
    'kwargs',
    [
        pytest.param(
            {},
            id='default',
        ),
        pytest.param(
            {'width_cm': None, 'height_cm': None},
            id='width-height-None',
        ),
        pytest.param(
            {'size': None},
            id='size-None',
        ),
        pytest.param(
            {'size': (None, None)},
            id='size-None-tuple',
        ),
    ],
)
def test_screen_init_size_is_none(kwargs):
    screen = pm.Screen(**kwargs)
    assert screen.size is None
    assert screen.width_cm is None
    assert screen.height_cm is None


@pytest.mark.parametrize(
    ('kwargs', 'expected_size'),
    [
        pytest.param(
            {'width_cm': 30.1},
            (30.1, None),
            id='width',
        ),
        pytest.param(
            {'height_cm': 27.8},
            (None, 27.8),
            id='height',
        ),
        pytest.param(
            {'width_cm': 30.2, 'height_cm': 27.6},
            (30.2, 27.6),
            id='width-height',
        ),
        pytest.param(
            {'size': (39.9, 32.1)},
            (39.9, 32.1),
            id='size',
        ),
        pytest.param(
            {'size': [38.7, 31.2]},
            (38.7, 31.2),
            id='size-list',
        ),
    ],
)
def test_screen_init_has_correct_size(kwargs, expected_size):
    screen = pm.Screen(**kwargs)
    assert isinstance(screen.size, tuple)
    assert screen.size == expected_size
    assert screen.width_cm == expected_size[0]
    assert screen.height_cm == expected_size[1]


@pytest.mark.parametrize(
    ('kwargs', 'exception', 'message'),
    [
        pytest.param(
            {'height_px': 768, 'resolution': (1024, 768)},
            ValueError,
            'The arguments "height_px" and "resolution" are mutually exclusive.',
            id='mutually_exclusive_height_px',
        ),
        pytest.param(
            {'width_px': 1024, 'resolution': (1024, 768)},
            ValueError,
            'The arguments "width_px" and "resolution" are mutually exclusive.',
            id='mutually_exclusive_width_px',
        ),
        pytest.param(
            {'height_cm': 30.1, 'size': (39.3, 30.1)},
            ValueError,
            'The arguments "height_cm" and "size" are mutually exclusive.',
            id='mutually_exclusive_height_cm',
        ),
        pytest.param(
            {'width_cm': 39.3, 'size': (39.3, 30.1)},
            ValueError,
            'The arguments "width_cm" and "size" are mutually exclusive.',
            id='mutually_exclusive_width_cm',
        ),
    ],
)
def test_screen_init_raises(kwargs, exception, message):
    with pytest.raises(exception, match=message):
        pm.Screen(**kwargs)


@pytest.mark.parametrize(
    ('name', 'value', 'exception', 'message'),
    [
        pytest.param(
            'height_px',
            'a',
            TypeError,
            "'height_px' must be a number but is of type str",
            id='height_px_str',
        ),
        pytest.param(
            'width_px',
            'a',
            TypeError,
            "'width_px' must be a number but is of type str",
            id='width_px_str',
        ),
        pytest.param(
            'height_cm',
            'a',
            TypeError,
            "'height_cm' must be a number but is of type str",
            id='height_cm_str',
        ),
        pytest.param(
            'width_cm',
            'a',
            TypeError,
            "'width_cm' must be a number but is of type str",
            id='width_cm_str',
        ),
        pytest.param(
            'height_px',
            -1,
            ValueError,
            "'height_px' must be a positive number but is: -1",
            id='height_px_negative',
        ),
        pytest.param(
            'width_px',
            -1,
            ValueError,
            "'width_px' must be a positive number but is: -1",
            id='width_px_negative',
        ),
        pytest.param(
            'height_cm',
            -1,
            ValueError,
            "'height_cm' must be a positive number but is: -1",
            id='height_cm_negative',
        ),
        pytest.param(
            'width_cm',
            -1,
            ValueError,
            "'width_cm' must be a positive number but is: -1",
            id='width_cm_negative',
        ),
        pytest.param(
            'height_px',
            0,
            ValueError,
            "'height_px' must be a positive number but is: 0",
            id='height_px_zero',
        ),
        pytest.param(
            'width_px',
            0,
            ValueError,
            "'width_px' must be a positive number but is: 0",
            id='width_px_zero',
        ),
        pytest.param(
            'height_cm',
            0,
            ValueError,
            "'height_cm' must be a positive number but is: 0",
            id='height_cm_zero',
        ),
        pytest.param(
            'width_cm',
            0,
            ValueError,
            "'width_cm' must be a positive number but is: 0",
            id='width_cm_zero',
        ),
        pytest.param(
            'resolution',
            1,
            TypeError,
            "'resolution' must be a sequence but is of type int",
            id='resolution_int',
        ),
        pytest.param(
            'resolution',
            (1,),
            ValueError,
            "'resolution' must be of length 2 but its length is 1",
            id='resolution_1d',
        ),
        pytest.param(
            'resolution',
            (1, 2, 3),
            ValueError,
            "'resolution' must be of length 2 but its length is 3",
            id='resolution_3d',
        ),
        pytest.param(
            'size',
            1,
            TypeError,
            "'size' must be a sequence but is of type int",
            id='size_int',
        ),
        pytest.param(
            'size',
            (1,),
            ValueError,
            "'size' must be of length 2 but its length is 1",
            id='size_1d',
        ),
        pytest.param(
            'size',
            (1, 2, 3),
            ValueError,
            "'size' must be of length 2 but its length is 3",
            id='size_3d',
        ),
    ],
)
def test_screen_set_attribute_raises(name, value, exception, message):
    screen = pm.Screen()
    with pytest.raises(exception, match=message):
        setattr(screen, name, value)


@pytest.mark.parametrize(
    ('value', 'expected_resolution', 'expected_width', 'expected_height'),
    [
        pytest.param(
            None,
            None,
            None,
            None,
            id='none',
        ),
        pytest.param(
            (1234, None),
            (1234, None),
            1234,
            None,
            id='width',
        ),
        pytest.param(
            (None, 3456),
            (None, 3456),
            None,
            3456,
            id='height',
        ),
        pytest.param(
            (1357, 2468),
            (1357, 2468),
            1357,
            2468,
            id='width_height',
        ),
    ],
)
def test_screen_set_resolution(value, expected_resolution, expected_width, expected_height):
    screen = pm.Screen()
    assert screen.width_px is None
    assert screen.height_px is None
    assert screen.resolution is None

    screen.resolution = value

    assert screen.width_px == expected_width
    assert screen.height_px == expected_height
    assert screen.resolution == expected_resolution


@pytest.mark.parametrize(
    ('value', 'expected_size', 'expected_width', 'expected_height'),
    [
        pytest.param(
            None,
            None,
            None,
            None,
            id='none',
        ),
        pytest.param(
            (1234, None),
            (1234, None),
            1234,
            None,
            id='width',
        ),
        pytest.param(
            (None, 3456),
            (None, 3456),
            None,
            3456,
            id='height',
        ),
        pytest.param(
            (1357, 2468),
            (1357, 2468),
            1357,
            2468,
            id='width_height',
        ),
    ],
)
def test_screen_set_size(value, expected_size, expected_width, expected_height):
    screen = pm.Screen()
    assert screen.width_cm is None
    assert screen.height_cm is None
    assert screen.size is None

    screen.size = value

    assert screen.width_cm == expected_width
    assert screen.height_cm == expected_height
    assert screen.size == expected_size


@pytest.mark.parametrize('property_name', ['x_max_dva', 'y_max_dva', 'x_min_dva', 'y_min_dva'])
def test_dva_properties_with_no_distance_cm(property_name):
    screen = pm.Screen(1920, 1080, 30, 20, None, 'upper left')
    with pytest.raises(TypeError):
        getattr(screen, property_name)


@pytest.mark.parametrize('property_name', ['x_max_dva', 'y_max_dva', 'x_min_dva', 'y_min_dva'])
def test_dva_properties_with_distance_cm(property_name):
    screen = pm.Screen(1920, 1080, 30, 20, 60, 'upper left')

    getattr(screen, property_name)


def test_screen_pix2deg_with_no_distance_cm():
    screen = pm.Screen(1920, 1080, 30, 20, None, 'upper left')
    with pytest.raises(TypeError):
        screen.pix2deg([[0, 0]])


def test_screen_pix2deg_with_distance_cm():
    screen = pm.Screen(1920, 1080, 30, 20, 60, 'upper left')
    screen.pix2deg([[0, 0]])


@pytest.mark.parametrize(
    ('missing_attribute', 'exception', 'exception_msg'),
    [
        pytest.param(
            'width_px',
            TypeError,
            "'width_px' must not be None",
            id='width_px',
        ),
        pytest.param(
            'height_px',
            TypeError,
            "'height_px' must not be None",
            id='height_px',
        ),
        pytest.param(
            'width_cm',
            TypeError,
            "'width_cm' must not be None",
            id='width_cm',
        ),
        pytest.param(
            'height_cm',
            TypeError,
            "'height_cm' must not be None",
            id='height_cm',
        ),
        pytest.param(
            'distance_cm',
            TypeError,
            "'distance_cm' must not be None",
            id='distance_cm',
        ),
        pytest.param(
            'origin',
            TypeError,
            "'origin' must not be None",
            id='origin',
        ),
    ],
)
def test_pix2deg_without_attributes(missing_attribute, exception, exception_msg):
    screen = pm.Screen(1920, 1080, 30, 20, 68.0, 'upper left')
    setattr(screen, missing_attribute, None)

    with pytest.raises(exception) as excinfo:
        screen.pix2deg([[0, 0]])

    msg, = excinfo.value.args
    assert msg == exception_msg


@pytest.mark.parametrize(
    ('screen', 'exclude_none', 'expected_dict'),
    [
        pytest.param(
            pm.Screen(),
            True,
            {},
            id='default',
        ),
        pytest.param(
            pm.Screen(height_px=150, origin='test'),
            True,
            {'height_px': 150, 'origin': 'test'},
            id='height_px_origin',
        ),
        pytest.param(
            pm.Screen(),
            False,
            {
                'width_px': None,
                'height_px': None,
                'width_cm': None,
                'height_cm': None,
                'distance_cm': None,
                'origin': None,
            },
            id='all_none',
        ),
    ],
)
def test_screen_to_dict_exclude_none(screen, exclude_none, expected_dict):
    assert screen.to_dict(exclude_none=exclude_none) == expected_dict


@pytest.mark.parametrize(
    ('prefer_resolution', 'expected_dict'),
    [
        pytest.param(
            True,
            {'resolution': (1024, 768)},
            id='True',
        ),
        pytest.param(
            False,
            {'width_px': 1024, 'height_px': 768},
            id='False',
        ),
    ],
)
def test_screen_to_dict_prefer_resolution(prefer_resolution, expected_dict):
    screen = pm.Screen(1024, 768)
    result = screen.to_dict(prefer_resolution=prefer_resolution, exclude_none=True)
    assert result == expected_dict


@pytest.mark.parametrize(
    ('prefer_size', 'expected_dict'),
    [
        pytest.param(
            True,
            {'size': (30.9, 27.1)},
            id='True',
        ),
        pytest.param(
            False,
            {'width_cm': 30.9, 'height_cm': 27.1},
            id='False',
        ),
    ],
)
def test_screen_to_dict_prefer_size(prefer_size, expected_dict):
    screen = pm.Screen(size=(30.9, 27.1))
    result = screen.to_dict(prefer_size=prefer_size, exclude_none=True)
    assert result == expected_dict


@pytest.mark.parametrize(
    ('screen', 'expected_bool'),
    [
        pytest.param(
            pm.Screen(),
            False,
            id='default',
        ),
        pytest.param(
            pm.Screen(origin=None),
            False,
            id='origin_none',
        ),
        pytest.param(
            pm.Screen(height_cm=10.0),
            True,
            id='height_cm_10',
        ),
        pytest.param(
            pm.Screen(width_px=300),
            True,
            id='width_px_300',
        ),
    ],
)
def test_screen_bool_all_none(screen, expected_bool):
    assert bool(screen) == expected_bool
