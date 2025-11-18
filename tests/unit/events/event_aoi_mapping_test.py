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
"""Test all Gaze functionality."""
import polars as pl
import pytest
from polars.testing import assert_frame_equal

import pymovements as pm

EXPECTED_DF = {
    'char': pl.DataFrame(
        [
            (
                0, 1, 'fixation', 1988147, 1988322, 175, 207.4090909090909, 151.54261363636363,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1988351, 1988546, 195, 167.10408163265308, 147.5826530612245,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1988592, 1988736, 144, 251.9013793103448, 152.1158620689655,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1988788, 1989013, 225, 375.60796460177, 156.98584070796457,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989060, 1989170, 110, 447.34954954954986, 153.80810810810806,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989202, 1989424, 222, 513.3, 157.17892376681613,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989461, 1989585, 124, 581.2648000000002, 159.66959999999992,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989640, 1989891, 251, 707.7650793650796, 163.16626984126978,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989929, 1990049, 120, 787.0628099173551, 163.24462809917353,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1990078, 1990304, 226, 846.2229074889868, 166.44008810572686,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1990356, 1990652, 296, 978.3326599326601, 163.44141414141413,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1990751, 1990926, 175, 180.84772727272727, 208.0573863636364,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1990961, 1991140, 179, 150.4911111111111, 211.21777777777777,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1991189, 1991317, 128, 239.65658914728684, 211.82480620155036,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1991353, 1991495, 142, 302.71608391608396, 216.57412587412588,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1991527, 1991715, 188, 375.24232804232804, 219.3873015873016,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1991745, 1991899, 154, 427.50838709677413, 217.13419354838712,
                'y', 414.972602739726, 214.85148514851485, 14.972602739726028, 23, 1, 1,
                'page_2', 'pymovements:', 429.94520547945206, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1991950, 1992230, 280, 600.1049822064056, 213.44768683274023,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1992284, 1992569, 285, 725.8087412587413, 222.21293706293707,
                ' ', 714.4246575342465, 214.85148514851485, 14.972602739726028, 23, 21, 1,
                'page_2', ' ', 729.3972602739725, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1992611, 1992757, 146, 791.1496598639455, 221.16598639455782,
                'a', 789.2876712328767, 214.85148514851485, 14.972602739726028, 23, 26, 1,
                'page_2', 'package', 804.2602739726028, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1992792, 1992969, 177, 848.5044943820226, 221.47415730337084,
                ' ', 834.2054794520548, 214.85148514851485, 14.972602739726028, 23, 29, 1,
                'page_2', ' ', 849.1780821917807, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1993004, 1993238, 234, 932.0889361702128, 221.67829787234044,
                'y', 924.0410958904106, 214.85148514851485, 14.972602739726028, 23, 35, 1,
                'page_2', 'eye', 939.0136986301366, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1993274, 1993510, 236, 1024.154852320675, 221.48607594936712,
                'e', 1013.8767123287664, 214.85148514851485, 14.972602739726028, 23, 41, 1,
                'page_2', 'movement', 1028.8493150684924, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1993610, 1993716, 106, 200.10841121495332, 264.7196261682243,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1993752, 1993948, 196, 157.92436548223353, 268.2375634517766,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1993982, 1994193, 211, 241.02405660377357, 271.5188679245283,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1994223, 1994383, 160, 296.44596273291927, 273.2776397515528,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1994436, 1994710, 274, 435.9218181818182, 286.8061818181818,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1994761, 1994900, 139, 522.06, 286.785, None,
                None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1994957, 1995109, 152, 606.8084967320262, 285.3562091503268,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1995156, 1995323, 167, 688.3916666666667, 284.9291666666666,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1995374, 1995532, 158, 770.340251572327, 289.7440251572327,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1995558, 1995805, 247, 825.5754032258062, 287.07620967741934,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1995841, 1995988, 147, 880.3533783783784, 283.3432432432432,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1996026, 1996198, 172, 938.5502890173411, 281.85953757225434,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1996245, 1996412, 167, 1022.4404761904764, 278.0517857142857,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1996517, 1996758, 241, 157.65289256198346, 323.6400826446281,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1996804, 1997014, 210, 223.56682464454977, 337.47156398104266,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997043, 1997190, 147, 274.86756756756756, 344.575,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997223, 1997438, 215, 368.6263888888888, 349.68055555555554,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997488, 1997653, 165, 460.4620481927711, 359.32289156626507,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997678, 1997905, 227, 432.2407894736842, 357.230701754386,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997954, 1998254, 300, 521.389368770764, 356.7348837209302,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1998285, 1998512, 227, 477.1824561403509, 357.9864035087719,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1998562, 1998790, 228, 577.8620087336244, 364.32532751091696,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1998838, 1999070, 232, 703.7060085836911, 360.56609442060096,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1999125, 1999382, 257, 830.8821705426357, 355.15271317829456,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1999436, 1999625, 189, 952.5884210526315, 352.0815789473684,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1999663, 1999819, 156, 1009.4426751592357, 342.1987261146497,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1999922, 2000103, 181, 157.89505494505497, 413.2324175824176,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000129, 2000245, 116, 184.99401709401718, 409.1675213675213,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000295, 2000455, 160, 272.43975155279503, 416.9608695652175,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000514, 2000641, 127, 389.371875, 417.5953125,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000677, 2000894, 217, 446.2220183486238, 419.33073394495403,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000930, 2001173, 243, 526.5098360655738, 423.3586065573772,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2001208, 2001459, 251, 612.4912698412701, 425.66468253968253,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2001495, 2001712, 217, 708.4536697247707, 423.444495412844,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2001746, 2001930, 184, 773.7962162162162, 423.02918918918914,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002003, 2002122, 119, 857.3725000000001, 422.345,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002160, 2002306, 146, 936.16462585034, 418.06870748299326,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002308, 2002442, 134, 940.3585185185185, 416.97555555555556,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002495, 2002603, 108, 1034.388990825688, 409.68256880733946,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002709, 2002831, 122, 134.0609756097561, 491.25121951219506,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002863, 2003003, 140, 161.38581560283689, 482.5148936170213,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2003042, 2003424, 382, 211.222454308094, 483.4208877284595,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2003453, 2003657, 204, 283.9468292682926, 485.65999999999997,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2003721, 2003917, 196, 422.41421319796956, 484.2720812182742,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2003968, 2004074, 106, 509.65233644859813, 480.192523364486,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2004132, 2004331, 199, 610.861, 484.6025000000001,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2004399, 2004687, 288, 717.8470588235293, 486.0269896193772,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2004714, 2004878, 164, 785.6884848484849, 481.4890909090909,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2004931, 2005109, 178, 896.4055865921787, 480.51899441340777,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2005138, 2005287, 149, 933.612, 481.8833333333333,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema=[
            'text_id',
            'page_id',
            'name',
            'onset',
            'offset',
            'duration',
            'location_x', 'location_y',
            'char',
            'top_left_x',
            'top_left_y',
            'width',
            'height',
            'char_idx_in_line',
            'line_idx',
            'page',
            'word',
            'bottom_left_x',
            'bottom_left_y',
        ],
        orient='row',
    ),
    'word': pl.DataFrame(
        [
            (
                0, 1, 'fixation', 1988147, 1988322, 175, 207.4090909090909, 151.54261363636363,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1988351, 1988546, 195, 167.10408163265308, 147.5826530612245,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1988592, 1988736, 144, 251.9013793103448, 152.1158620689655,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1988788, 1989013, 225, 375.60796460177, 156.98584070796457,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989060, 1989170, 110, 447.34954954954986, 153.80810810810806,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989202, 1989424, 222, 513.3, 157.17892376681613,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989461, 1989585, 124, 581.2648000000002, 159.66959999999992,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989640, 1989891, 251, 707.7650793650796, 163.16626984126978,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1989929, 1990049, 120, 787.0628099173551, 163.24462809917353,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1990078, 1990304, 226, 846.2229074889868, 166.44008810572686,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1990356, 1990652, 296, 978.3326599326601, 163.44141414141413,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1990751, 1990926, 175, 180.84772727272727, 208.0573863636364,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1990961, 1991140, 179, 150.4911111111111, 211.21777777777777,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1991189, 1991317, 128, 239.65658914728684, 211.82480620155036,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1991353, 1991495, 142, 302.71608391608396, 216.57412587412588,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1991527, 1991715, 188, 375.24232804232804, 219.3873015873016,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1991745, 1991899, 154, 427.50838709677413, 217.13419354838712,
                'y', 414.972602739726, 214.85148514851485, 14.972602739726028, 23, 1, 1,
                'page_2', 'pymovements:', 429.94520547945206, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1991950, 1992230, 280, 600.1049822064056, 213.44768683274023,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1992284, 1992569, 285, 725.8087412587413, 222.21293706293707,
                ' ', 714.4246575342465, 214.85148514851485, 14.972602739726028, 23, 21, 1,
                'page_2', ' ', 729.3972602739725, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1992611, 1992757, 146, 791.1496598639455, 221.16598639455782,
                'a', 789.2876712328767, 214.85148514851485, 14.972602739726028, 23, 26, 1,
                'page_2', 'package', 804.2602739726028, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1992792, 1992969, 177, 848.5044943820226, 221.47415730337084,
                ' ', 834.2054794520548, 214.85148514851485, 14.972602739726028, 23, 29, 1,
                'page_2', ' ', 849.1780821917807, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1993004, 1993238, 234, 932.0889361702128, 221.67829787234044,
                'y', 924.0410958904106, 214.85148514851485, 14.972602739726028, 23, 35, 1,
                'page_2', 'eye', 939.0136986301366, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1993274, 1993510, 236, 1024.154852320675, 221.48607594936712,
                'e', 1013.8767123287664, 214.85148514851485, 14.972602739726028, 23, 41, 1,
                'page_2', 'movement', 1028.8493150684924, 237.85148514851485,
            ),
            (
                0, 1, 'fixation', 1993610, 1993716, 106, 200.10841121495332, 264.7196261682243,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1993752, 1993948, 196, 157.92436548223353, 268.2375634517766,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1993982, 1994193, 211, 241.02405660377357, 271.5188679245283,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1994223, 1994383, 160, 296.44596273291927, 273.2776397515528,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1994436, 1994710, 274, 435.9218181818182, 286.8061818181818,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1994761, 1994900, 139, 522.06, 286.785, None,
                None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1994957, 1995109, 152, 606.8084967320262, 285.3562091503268,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1995156, 1995323, 167, 688.3916666666667, 284.9291666666666,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1995374, 1995532, 158, 770.340251572327, 289.7440251572327,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1995558, 1995805, 247, 825.5754032258062, 287.07620967741934,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1995841, 1995988, 147, 880.3533783783784, 283.3432432432432,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1996026, 1996198, 172, 938.5502890173411, 281.85953757225434,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1996245, 1996412, 167, 1022.4404761904764, 278.0517857142857,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1996517, 1996758, 241, 157.65289256198346, 323.6400826446281,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1996804, 1997014, 210, 223.56682464454977, 337.47156398104266,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997043, 1997190, 147, 274.86756756756756, 344.575,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997223, 1997438, 215, 368.6263888888888, 349.68055555555554,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997488, 1997653, 165, 460.4620481927711, 359.32289156626507,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997678, 1997905, 227, 432.2407894736842, 357.230701754386,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1997954, 1998254, 300, 521.389368770764, 356.7348837209302,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1998285, 1998512, 227, 477.1824561403509, 357.9864035087719,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1998562, 1998790, 228, 577.8620087336244, 364.32532751091696,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1998838, 1999070, 232, 703.7060085836911, 360.56609442060096,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1999125, 1999382, 257, 830.8821705426357, 355.15271317829456,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1999436, 1999625, 189, 952.5884210526315, 352.0815789473684,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1999663, 1999819, 156, 1009.4426751592357, 342.1987261146497,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 1999922, 2000103, 181, 157.89505494505497, 413.2324175824176,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000129, 2000245, 116, 184.99401709401718, 409.1675213675213,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000295, 2000455, 160, 272.43975155279503, 416.9608695652175,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000514, 2000641, 127, 389.371875, 417.5953125,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000677, 2000894, 217, 446.2220183486238, 419.33073394495403,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2000930, 2001173, 243, 526.5098360655738, 423.3586065573772,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2001208, 2001459, 251, 612.4912698412701, 425.66468253968253,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2001495, 2001712, 217, 708.4536697247707, 423.444495412844,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2001746, 2001930, 184, 773.7962162162162, 423.02918918918914,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002003, 2002122, 119, 857.3725000000001, 422.345,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002160, 2002306, 146, 936.16462585034, 418.06870748299326,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002308, 2002442, 134, 940.3585185185185, 416.97555555555556,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002495, 2002603, 108, 1034.388990825688, 409.68256880733946,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002709, 2002831, 122, 134.0609756097561, 491.25121951219506,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2002863, 2003003, 140, 161.38581560283689, 482.5148936170213,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2003042, 2003424, 382, 211.222454308094, 483.4208877284595,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2003453, 2003657, 204, 283.9468292682926, 485.65999999999997,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2003721, 2003917, 196, 422.41421319796956, 484.2720812182742,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2003968, 2004074, 106, 509.65233644859813, 480.192523364486,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2004132, 2004331, 199, 610.861, 484.6025000000001,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2004399, 2004687, 288, 717.8470588235293, 486.0269896193772,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2004714, 2004878, 164, 785.6884848484849, 481.4890909090909,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2004931, 2005109, 178, 896.4055865921787, 480.51899441340777,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
            (
                0, 1, 'fixation', 2005138, 2005287, 149, 933.612, 481.8833333333333,
                None, None, None, None, None, None, None, None, None, None, None,
            ),
        ],
        schema=[
            'text_id',
            'page_id',
            'name',
            'onset',
            'offset',
            'duration',
            'location_x', 'location_y',
            'char',
            'top_left_x',
            'top_left_y',
            'width',
            'height',
            'char_idx_in_line',
            'line_idx',
            'page',
            'word',
            'bottom_left_x',
            'bottom_left_y',
        ],
        orient='row',
    ),
}


@pytest.fixture(name='dataset')
def dataset_fixture(tmp_path):
    dataset = pm.Dataset('ToyDataset', tmp_path)
    dataset.download()
    dataset.load()
    dataset.pix2deg()
    dataset.pos2vel()
    dataset.detect_events('ivt')
    dataset.compute_event_properties(('location', {'position_column': 'pixel'}))
    yield dataset


@pytest.mark.parametrize(
    ('aoi_column'),
    [
        'word',
        'char',
    ],
)
def test_event_to_aoi_mapping_char_width_height(aoi_column, dataset, make_example_file):
    filepath = make_example_file('toy_text_1_1_aoi.csv')

    aoi_df = pm.stimulus.text.from_file(
        filepath,
        aoi_column=aoi_column,
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )

    dataset.events[0].map_to_aois(aoi_df)
    assert_frame_equal(dataset.events[0].frame, EXPECTED_DF[aoi_column])


@pytest.mark.parametrize(
    ('aoi_column'),
    [
        'word',
        'char',
    ],
)
def test_event_to_aoi_mapping_char_end(aoi_column, dataset, make_example_file):
    filepath = make_example_file('toy_text_1_1_aoi.csv')

    aoi_df = pm.stimulus.text.from_file(
        filepath,
        aoi_column=aoi_column,
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        end_x_column='bottom_left_x',
        end_y_column='bottom_left_y',
        page_column='page',
    )

    dataset.events[0].map_to_aois(aoi_df)
    assert_frame_equal(dataset.events[0].frame, EXPECTED_DF[aoi_column])


def test_map_to_aois_raises_value_error(make_example_file):
    aoi_filepath = make_example_file('toy_text_1_1_aoi.csv')
    gaze_filepath = make_example_file('judo1000_example.csv')

    aoi_df = pm.stimulus.text.from_file(
        aoi_filepath,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        width_column='width',
        height_column='height',
        page_column='page',
    )
    gaze = pm.gaze.io.from_csv(
        gaze_filepath,
        read_csv_kwargs={'separator': '\t'},
        position_columns=['x_left', 'y_left', 'x_right', 'y_right'],
    )

    with pytest.raises(ValueError) as excinfo:
        gaze.events.map_to_aois(aoi_df)
    msg, = excinfo.value.args
    assert msg == 'cannot concat empty list'


def test_map_to_aois_raises_value_error_missing_width_height(dataset, make_example_file):
    filepath = make_example_file('toy_text_1_1_aoi.csv')

    aoi_df = pm.stimulus.text.from_file(
        filepath,
        aoi_column='char',
        start_x_column='top_left_x',
        start_y_column='top_left_y',
        page_column='page',
    )
    with pytest.raises(ValueError) as excinfo:
        dataset.events[0].map_to_aois(aoi_df)
    msg, = excinfo.value.args
    assert msg == 'either TextStimulus.width or TextStimulus.end_x_column must be defined'
