"""問題1の後半（プログラムによる確認）"""
import numpy as np

rs = np.random.RandomState(42)
n = 4
B = rs.rand(n, n)
print(B)
"""
[[0.37454012 0.95071431 0.73199394 0.59865848]
 [0.15601864 0.15599452 0.05808361 0.86617615]
 [0.60111501 0.70807258 0.02058449 0.96990985]
 [0.83244264 0.21233911 0.18182497 0.18340451]]
"""

A = np.dot(B.T, B)
print(A)
"""
[[1.21892212 0.98281189 0.44695573 1.09506235]
 [0.98281189 1.47464666 0.75816171 1.42998244]
 [0.44695573 0.75816171 0.57267288 0.54183765]
 [1.09506235 1.42998244 0.54183765 2.08301543]]
"""


for _ in range(5):
    x = rs.rand(n)
    xTAx = np.dot(x, np.dot(A, x))
    is_nonnegative = (xTAx >= 0)
    print(f"{xTAx=}, {is_nonnegative=}")
    """
    xTAx=2.3448553824012173, is_nonnegative=True
    xTAx=1.9559046849520527, is_nonnegative=True
    xTAx=4.538994617211748, is_nonnegative=True
    xTAx=1.4777079672301598, is_nonnegative=True
    xTAx=7.949924623318303, is_nonnegative=True
    """
