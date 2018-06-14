#!/usr/bin/env python
# -*- coding: utf-8 -*-
##
##  predictxx.py
##  refer ...
##  arranged by y.m. 2017/8/13-
##  rel 01
##  predictxx
## 		1 input-file
##		2 Pair ClGBPUSD...
##		3 0(silent)/1
##		4 period(1/2/3/4/5/6/7/10/0) MN1,W1,D1,H4,/20
##		5 multiplier(1/2/3/10/etc)
##

import pandas as pd
import sys

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# on UTF-8 using cp932 
df = pd.read_csv(sys.argv[1] , encoding="cp932")
df = df.sort_values(by=["index"], ascending=True)

# argv2 pair close

clpair = sys.argv[2]

# argv3 means print out 0...none else...print
if sys.argv[3] == str("0"):
 	isi = 0
else:
	isi = 1

if isi == str("1"):
 	print(df.tail(20))

#argv4 means period 1...month 2...week 3...day 4...4H 5...1H 6...30M
if sys.argv[4] == str("1"):
 	iper = 1
elif sys.argv[4] == str("2"):
	iper = 2
elif sys.argv[4] == str("3"):
	iper = 3
elif sys.argv[4] == str("4"):
	iper = 4
elif sys.argv[4] == str("5"):
	iper = 5
elif sys.argv[4] == str("10"):
	iper = 10
elif sys.argv[4] == str("20"):
	iper = 10
else:
	iper = 0

#fm means multiplier 1...EURUSD etc 2...JPY Pairs 3...Bitcoins else...JPN225
if sys.argv[5] == str("1"):
 	fm = 10000
elif sys.argv[5] == str("2"):
	fm = 100
elif sys.argv[5] == str("3"):
	fm = 10
elif sys.argv[5] == str("10"):
	fm = 0.5
else:
	fm = 1



df = df.iloc[0:len(df) - 1]
if isi == str("1"):
	print(df.tail())

df_train = df.iloc[1:len(df)-1]	# 最新以外
df_test = df.iloc[len(df)-1:len(df)] #最新

##print("AAA train", df_train)
##print("BB tBest", df_test)

xlist1 = [
	"DfEURUSD",
	"DfGBPUSD",
	"DfAUDUSD",
	"DfNZDUSD",
	"DfEURGBP",
	"DfEURAUD",
	"DfEURNZD",
	"DfGBPAUD",
	"DfGBPNZD",
	"DfAUDNZD",
	"DfUSDJPY",
	"DfAUDJPY",
	"DfEURJPY",
	"DfGBPJPY",
	"DfNZDJPY",
	"DfXAUUSD",
	"DfOILUSD",
	"DfJPN225",
	"DfUSA.30",
	]

xlist2 = [
	"DfEURUSD",
	"DfGBPUSD",
	"DfAUDUSD",
	"DfNZDUSD",
	"DfEURGBP",
	"DfEURAUD",
	"DfEURNZD",
	"DfGBPAUD",
	"DfGBPNZD",
	"DfAUDNZD",
	"DfUSDJPY",
	"DfAUDJPY",
	"DfEURJPY",
	"DfGBPJPY",
	"DfNZDJPY",
	"DfXAUUSD",
	"DfOILUSD",
	"DfJPN225",
	"DfUSA.30",
	]


xlist= xlist1
#if iper== 3: # Day
#	xlist = xlist1
#elif iper == 4: # 4H
#	xlist = xlist1
#elif iper == 2: # Week
#	xlist = xlist1
#elif iper == 1: # Week
#	xlist = xlist1
#elif iper == 10: # Day2
#	xlist = xlist1
#elif iper == 20: # Day2
#	xlist = xlist1

x_train = []
y_train = []

for s in range(0, len(df_train) - 1):
	if isi == int("1"):
		print("x_train : ", df_train["Date"].iloc[s])
		print("y_train : ", df_train["Date"].iloc[s + 1])
		print("")
	x_train.append(df_train[xlist].iloc[s]) 


	if iper== int("3"):
		df1 = (df_train[clpair].iloc[s + 1] - df_train[clpair].iloc[s] )*fm
		if isi == int("1"):
			print( df1 )
		iap = 0
		if df1 < -50:
			iap = -5
		elif df1 < -40:
			iap = -4
		elif df1 < -30:
			iap = -3
		elif df1 < -20:
			iap = -2
		elif df1 < -10:
			iap = -1
		elif df1 > 50:
			iap = 5
		elif df1 > 40:
			iap = 4
		elif df1 > 30:
			iap = 3
		elif df1 > 20:
			iap = 2
		elif df1 > 10:
			iap = 1

	elif iper== int("4"):
		df1 = (df_train[clpair].iloc[s + 1] - df_train[clpair].iloc[s] )*fm
		if isi == int("1"):
			print( df1 )
		iap = 0
		if df1 < -25:
			iap = -5
		elif df1 < -20:
			iap = -4
		elif df1 < -15:
			iap = -3
		elif df1 < -10:
			iap = -2
		elif df1 < -5:
			iap = -1
		elif df1 > 25:
			iap = 5
		elif df1 > 20:
			iap = 4
		elif df1 > 15:
			iap = 3
		elif df1 > 10:
			iap = 2
		elif df1 > 5:
			iap = 1
	elif iper == int("2") or iper == int("1"):
		df1 = (df_train[clpair].iloc[s + 1] - df_train[clpair].iloc[s] )*fm
		#print( df1 )
		iap = 0
		iap = int(df1 / 50)

	elif iper == int("10"):
		df1 = (df_train[clpair].iloc[s + 1] - df_train[clpair].iloc[s] )*fm
		#print( df1 )
		iap = 0
		iap = int(df1 / 10)
	elif iper == int("10"):
		df1 = (df_train[clpair].iloc[s + 1] - df_train[clpair].iloc[s] )*fm
		#print( df1 )
		iap = 0
		iap = int(df1 / 20)



	#if df_train[clpair].iloc[s + 1] > df_train[clpair].iloc[s]:
	#	y_train.append(1)
	#else:
	#	y_train.append(-1)
	y_train.append(iap)


if isi == int("1"):
	print(x_train)
	print(y_train)

rf = RandomForestClassifier(n_estimators=len(x_train), random_state=0)
#rf = RandomForestClassifier(10, random_state=0)
rf.fit(x_train, y_train)


test_x = df_test[xlist].iloc[0]
test_y = rf.predict(test_x.reshape(1, -1))
# output predict result
print("result : ", test_y[0])
