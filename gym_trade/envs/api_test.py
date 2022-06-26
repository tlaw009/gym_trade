import yfinance as yf
import numpy as np
import re

msft = yf.Ticker("MSFT")

vix = yf.Ticker("^VIX")

def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """

    h = array.shape[0]
    w = array.shape[1]

    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((a, aa), (b, bb)), mode='constant')



# print(vix.history(period="5d", interval="1d"))
# print(msft.history(period="5d", interval="1d"))
# print(msft.history(period="5d", interval="1d"))
# print(msft.history(period="1d", interval="1m"))
# print(msft.history(period="1d", interval="1m").to_numpy()[-2][4])
# print(msft.history(period="1d", interval="1m").to_numpy()[-1,0])

# print(vix.history(period="5d", interval="1d").to_numpy().shape)
# print(msft.history(period="5d", interval="1d").to_numpy().shape)

# print(msft.balance_sheet
#     )

# print(msft.balance_sheet.to_numpy().shape
#     )
try:
    data1 = yf.download("WISA", start = "2019-01-01", end = "2022-02-21", group_by = "ticker")
except:
    raise Exception("download fail, dates invalid")

print(data1.to_numpy()[0][5])
# data2 = msft.history(period="5d", interval="1d")
# print(data2)
# print(np.delete(data2.to_numpy(), [-2,-1], 1))
# print(padding(np.delete(data1.to_numpy(), 4 , 1), 26, 5))
# print(padding(np.delete(data1.to_numpy()[0:5], 4 , 1), 35, 5))
# print(msft.balance_sheet)
# print(data1.to_numpy()[0][1])
# print(msft.history(period="5d", interval="1d"))

# print(padding(msft.balance_sheet.to_numpy(), 26,7))