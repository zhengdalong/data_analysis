import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def pandas_basic():
	s = pd.Series([1,3,6,np.nan,44,1])
	print(s)
	dates = pd.date_range('20181010',periods=6)
	print(dates)
	df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
	print(df)
	df = pd.DataFrame(np.arange(12).reshape((3,4)))
	print(df)
	dic = {}
	dic['a'] = 1
	dic['b'] = pd.Timestamp('20130102')
	dic['c'] = pd.Series(1,index=list(range(4)),dtype='float32')
	dic['d'] = np.array([3]*4,dtype='int32')
	dic['e'] = pd.Categorical(['test','train','test','train'])
	dic['f'] = 'foo'
	df = pd.DataFrame(dic)
	print(df)
	print(df.dtypes)
	print(df.index)
	print(df.columns)
	print(df.values)
	print(df.describe())
	print(df.T)
	print(df.sort_index(axis=1,ascending=False))
	print(df.sort_values(by='e'))
	pass


def pandas_data_choose():
	dates = pd.date_range('20181010',periods=6)
	df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['a','b','c','d'])
	print(df)
	print(df['a'],df.a)
	print(df[0:3],df['20181010':'20181013'])
	print(df.loc['20181010'])
	print(df.loc[:,['a','b']])
	print(df.loc['20181010',['a','b']])
	print(df.iloc[3:5,1:3])
	print(df.iloc[[1,3,5],1:3])
	print(df.ix[[1,3,5],['a','c']])
	print(df[df.a>7])
	pass


def pandas_set():
	dates = pd.date_range('20181010',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)*10,index=dates,columns=['a','b','c','d'])
	print(df)
	df.iloc[2,2] = 1111
	print(df)
	df.loc['20181010','a'] = 2222
	print(df)
	df.a[df.a>2] = 0
	print(df)
	df.b[df.a>2] = 0
	print(df)
	df['e'] = np.nan
	print(df)
	df['f'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20181010',periods=6))
	print(df)
	pass


def pandas_nan():
	dates = pd.date_range('20181010',periods=6)
	df = pd.DataFrame(np.random.randn(6,4)*10,index=dates,columns=['a','b','c','d'])
	df.iloc[0,1] = np.nan
	df.iloc[1,2] = np.nan
	print(df)
	print(df.dropna(axis=0,how='all'))
	print(df.fillna(value=0))
	print(df.isnull())
	print(np.any(df.isnull())==True)
	pass


def pandas_concat():
	df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
	df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
	df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'])
	print(df1)
	print(df2)
	print(df3)
	res = pd.concat([df1,df2,df3],axis=0,ignore_index=True)
	print(res)

	df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'],index=[1,2,3])
	df2 = pd.DataFrame(np.ones((3,4))*1,columns=['b','c','d','e'],index=[2,3,4])
	print(df1)
	print(df2)
	res = pd.concat([df1,df2],join='outer')
	print(res)
	res = pd.concat([df1,df2],join='inner')
	print(res)
	res = pd.concat([df1,df2],axis=1,join_axes=[df1.index])
	print(res)

	df1 = pd.DataFrame(np.ones((3,4))*0,columns=['a','b','c','d'])
	df2 = pd.DataFrame(np.ones((3,4))*1,columns=['a','b','c','d'])
	df3 = pd.DataFrame(np.ones((3,4))*2,columns=['a','b','c','d'],index=[2,3,4])
	res = df1.append(df2,ignore_index=True)
	res = df1.append([df2,df3])

	s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
	res = df1.append(s1,ignore_index=True)
	print(res)
	pass

def pandas_merge():
	left = {}
	left['key'] = ['k0','k1','k2','k3']
	left['ke1'] = ['k0','k1','k2','k3']
	left['ke2'] = ['k0','k1','k2','k3']
	left['a'] = ['a0','a1','a2','a3']
	left['b'] = ['b0','b1','b2','b3']
	right = {}
	right['key'] = ['k0','k1','k2','k3']
	right['key1'] = ['k0','k1','k2','k3']
	right['key2'] = ['k0','k1','k2','k3']
	right['c'] = ['c0','c1','c2','c3']
	right['d'] = ['d0','d1','d2','d3']
	le = pd.DataFrame(left)
	ri = pd.DataFrame(right)
	print(le)
	print(ri)
	res = pd.merge(le,ri,on='key')
	print(res)
	#res = pd.merge(le,ri,on=['key1','key2'],how='inner')
	print(res)
	df1 = pd.DataFrame({'col1':[0,1],'col_left':['a','b']})
	df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
	print(df1)
	print(df2)
	res = pd.merge(df1,df2,on='col1',how='outer',indicator=True)
	print(res)

	left = pd.DataFrame({'a':['a0','a1','a2'],
		                 'b':['b0','b1','b2']},index=['k0','k1','k2'])
	right = pd.DataFrame({'c':['c0','c1','c2'],
		                 'd':['d0','d1','d2']},index=['k0','k2','k3'])
	print(left)
	print(right)
	res = pd.merge(left,right,left_index=True,right_index=True,how='outer')
	print(res)

	boys = pd.DataFrame({'k':['k0','k1','k2'],'age':[1,2,3]})
	girls = pd.DataFrame({'k':['k0','k1','k2'],'age':[4,5,6]})
	res = pd.merge(boys,girls,on='k',suffixes=['_boy','_girl'],how='inner')
	print(res)
	pass


def pandas_plot():
	data = pd.Series(np.random.randn(1000),index=np.arange(1000))
	data = data.cumsum()
	#data.plot()
	#plt.show()

	data = pd.DataFrame(np.random.randn(1000,4),index=np.arange(1000),columns=list('abcd'))
	print(data.head())
	data = data.cumsum()
	print(data)
	#plot methods:
	# 'bar','hist','box','kde','area','scatter','hexbin','pie'
	ax = data.plot.scatter(x='a',y='b',color='darkblue',label='class 1')
	data.plot.scatter(x='a',y='c',color='darkgreen',label='class 2',ax=ax)
	pass


#pandas_basic()
#pandas_data_choose()
#pandas_set()
#pandas_nan()
#pandas_concat()
#pandas_merge()
pandas_plot()





























