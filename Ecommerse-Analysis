#Installation
pip install pandas numpy geopy matplotlib folium -U scikit-learn seaborn
 
#Importing
import seaborn as sns
%matplotlib
import matplotlib.pyplot as plt
import calmap
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
 
#Modules
data = pd.read_excel(r"E:\Users\ADMIN\Downloads\SET\data.xls")# Data set 
dels = data
temp = data
temp.drop(['Order ID','Postal Code','Country','Product ID'],axis=1,inplace = True)
main_x = temp.loc[:,'Row ID':'Product Name':1]
main_y = temp.loc[:,'Sales':'Profit':1]
train_x,test_x,train_y,test_y = train_test_split(main_x,main_y,test_size = 0.3)
train_d,test_d = train_test_split(temp,test_size = 0.3)

#  Shipping Mode stat (additions)
mode = dict(temp['Ship Mode'].value_counts())
dtlist = list(mode.keys())
mt = list(mode.values())
fig,ax = plt.subplots(figsize=(16, 9))
plt.barh(dtlist,mt)
for i,j in zip(mt,dtlist):
    plt.text(i,j,str(i))
plt.title('Shipping Classes Observation', fontdict={'size':36})
plt.show()

# Product type sales in percentage :1
fig, (ax1,ax2) = plt.subplots(1,2,figsize=(19,26))
wedges2, texts2, autotexts2 = ax2.pie(train_d['Sub-Category'].value_counts(),labels = train_d['Sub-Category'].unique(),
                                  autopct="%1.1f%%", shadow=False)
wedges1, texts1, autotexts1 = ax1.pie(train_d['Category'].value_counts(),labels = train_d['Category'].unique(),
                                  autopct="%1.1f%%", shadow=False)
ax1.legend(wedges1,train_d['Category'].unique(),title ='Main Category',
          loc ="upper left",
          bbox_to_anchor =(1, 0.15, 1, 1))
ax2.legend(wedges2,train_d['Sub-Category'].unique(),title ='product(Sub Category)',
          loc ="upper left",
          bbox_to_anchor =(1, 0.15, 1, 1))
plt.title('Sales Observation of Product-type', fontdict={'size':26})
plt.show()


# new data seperations
dt = dels[['Order Date','Category','Sub-Category','Sales','Discount','Profit','Quantity','Product Name']]


# profitabel producttype oover the years :2
def newprofit(X):#overall profit in last years
    datas = []
    pro = X['Profit'].mean()
    sal = X['Sales'].mean()
    
    pro_per = ((pro*100)/(sal-pro))
    datas.append(X['year'].unique())
    datas.append(*X['month'].unique())
    datas.append(X['Quantity'].sum())
    datas.append(sal)
    datas.append(X['Sales'].sum())
    datas.append(pro)
    datas.append(X['Profit'].sum())
    datas.append(round(pro_per))
    return pd.Series(datas,index=['year','Month','ovl_Quantity','ovl_Avg_Sales','ovl_Total_Sales','ovl_Avg_Profit','ovl_Total_profit','ovl_Profit_Precentage'])


def productdetails(x):
    d = []
    d.append(x['Profit'].sum())
    d.append(x['Sales'].sum())
    d.append(x['Profit'].mean())
    d.append(x['Product Name'].unique())
    d.append(x['Quantity'].sum())
    d.append(*x['Category'].unique())
    return pd.Series(d, index=['Profit','Total_Sales','Average Profit % gained','Products','Quantity','category'])


def grossprofit(X):
    datas = []
    pro = X['Profit'].mean()
    sal = X['Sales'].mean()
    pro_per = ((pro*100)/(sal-pro))
    datas.append(*X['year'].unique())
    datas.append(*X['month'].unique())
    datas.append(X['Quantity'].sum())
    datas.append(sal)
    datas.append(X['Sales'].sum())
    datas.append(pro)
    datas.append(X['Profit'].sum())
    datas.append(round(pro_per))
    return pd.Series(datas,index=['year','Month','Quantity','Avg_Sales','Total_Sales','Avg_Profit','Total_profit','Profit_Precentage'])


dt['Date'] = pd.to_datetime(dt['Order Date'])
dt['year'], dt['month'] = dt['Date'].dt.year, dt['Date'].dt.month
dt = dt.sort_values(['Profit'], ascending=False)
yr = sorted(list(dt['year'].unique()))
categories = np.unique(dt['Category'])
colorss = [plt.cm.tab10(i/float(len(categories)-1)) for i in range(len(categories))]

for i in yr:
    #Profitable Product-types 
    
    pro = dt[dt['year'] == i ]
    Sub_Categorygrp = pro.groupby('Sub-Category').apply(productdetails)
    productgrp = pro.groupby('Product Name').apply(productdetails)
    Sub_Categorygrp =  Sub_Categorygrp.sort_values('Profit',ascending=False)
    productgrp =  productgrp.sort_values('Profit',ascending=False)
    x = Sub_Categorygrp.loc[:, ['Profit']]
    Sub_Categorygrp['Profit_z'] = (x - x.mean())/x.std()
    Sub_Categorygrp['colors'] = ['red' if x < 0 else 'green' for x in Sub_Categorygrp['Profit']]
    Sub_Categorygrp.sort_values('Profit_z', inplace=True)

    plt.figure(figsize=(20,16), dpi= 80)
    plt.hlines(y=Sub_Categorygrp.index, xmin=0, xmax=Sub_Categorygrp['Profit'], color=Sub_Categorygrp.colors,linewidth=5)
    for x, y, tex in zip(Sub_Categorygrp['Profit'], Sub_Categorygrp.index, Sub_Categorygrp['Profit']):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if x < 0 else 'left', 
                 verticalalignment='center', fontdict={'color':'red' if x < 0 else 'green', 'size':10})

    plt.gca().set(ylabel='$Product$', xlabel='$Profit$')
    plt.yticks(Sub_Categorygrp.index, fontsize=12)
    plt.title('Profitable of Product Type in {}'.format(i), fontdict={'size':36})
    plt.grid(linestyle='--', alpha=0.4)
    plt.show()
    
#     most number of quantity observations : 3
    midwest_encircle_data = dt.loc[dt['year']== i, :] 
    midwest_encircle_data = midwest_encircle_data.sort_values('Sub-Category',ascending=True)
    plt.figure(figsize=(20,16), dpi= 80)
    sns.violinplot(x='Sub-Category',y='Quantity',data=midwest_encircle_data,scale='width', inner='quartile')
    plt.xticks(fontsize=12); plt.yticks(fontsize=12)
    plt.title("Frequency distribution of quantity in {}".format(i), fontsize=36)
    plt.show()
    
#     Monthly profit observations :4
    fig , ax = plt.subplots(figsize=(20,16), dpi= 80)
    nx = dt[dt['year'] == i]
    profitsales = nx.groupby('month').apply(grossprofit)
    profitsales = profitsales.sort_values('month',ascending=True)
    sns.set_style("white")
    sns.regplot(x="Month", y="Profit_Precentage",data=profitsales,fit_reg=True,logx=True,ax = ax,line_kws=dict(label='Predicted line'),scatter_kws=dict(s=100, linewidths=.7, edgecolors='black'))
    plt.plot(profitsales['Month'], profitsales['Profit_Precentage'], '-o', color='red',linewidth=1.5,label='actual Value')
    for x, y, tex in zip(profitsales['Month'], profitsales['Profit_Precentage'], profitsales['Profit_Precentage']):
        t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if y < 0 else 'left', 
                 verticalalignment='top', fontdict={'color':'red' if y < 0 else 'green', 'size':20})
    plt.grid(linestyle='--', alpha=0.4)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=20)
    plt.title("Yearly Profit Observations in {}".format(i), fontsize=36)
    plt.show()
    
#overall profit in last years
my = dt.groupby(['month']).apply(newprofit)
my = my.sort_values('month',ascending=True)
fig , ax = plt.subplots(figsize=(20,16), dpi= 80)
sns.set_style("white")
sns.regplot(x="Month", y="ovl_Profit_Precentage",data=my,fit_reg=True,logx=True,ax = ax,line_kws=dict(label='Predicted line',color='red'),scatter_kws=dict(s=100, linewidths=.7, edgecolors='black'))
plt.plot(my['Month'], my['ovl_Profit_Precentage'], '-o', color='purple',linewidth=1.5,label='actual Value')
for x, y, tex in zip(my['Month'], my['ovl_Profit_Precentage'], my['ovl_Profit_Precentage']):
    t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if y < 0 else 'left', 
             verticalalignment='top', fontdict={'color':'red' if y < 0 else 'green', 'size':20})
plt.grid(linestyle='--', alpha=0.4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=20)
plt.title("Overall Profit Observations", fontsize=36)
plt.show()

    
#overall profit in last years
my = dt.groupby(['month']).apply(newprofit)
my = my.sort_values('month',ascending=True)
fig , ax = plt.subplots(figsize=(20,16), dpi= 80)
sns.set_style("white")
sns.regplot(x="Month", y="ovl_Profit_Precentage",data=my,fit_reg=True,logx=True,ax = ax,line_kws=dict(label='Predicted line',color='red'),scatter_kws=dict(s=100, linewidths=.7, edgecolors='black'))
plt.plot(my['Month'], my['ovl_Profit_Precentage'], '-o', color='purple',linewidth=1.5,label='actual Value')
for x, y, tex in zip(my['Month'], my['ovl_Profit_Precentage'], my['ovl_Profit_Precentage']):
    t = plt.text(x, y, round(tex, 2), horizontalalignment='right' if y < 0 else 'left', 
             verticalalignment='top', fontdict={'color':'red' if y < 0 else 'green', 'size':20})
plt.grid(linestyle='--', alpha=0.4)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(fontsize=20)
plt.title("Overall Profit Observations", fontsize=36)
plt.show()

#Product type that was ordered greater times
import random 
def product_type_has_greaterorders(x):
    d = []
    d.append(x['Quantity'].sum())
    d.append(x['Sales'].sum())
    d.append(*x['Sub-Category'].unique())
    return pd.Series(d,index=['Ordered','Sales','Product_type'])
prdGO = dels.groupby("Sub-Category").apply(product_type_has_greaterorders)
plt.figure(figsize=(20,16), dpi= 80)
n = prdGO['Product_type'].unique().__len__()+1
all_colors = list(plt.cm.colors.cnames.keys())
random.seed(100)
c = random.choices(all_colors, k=n)
plt.bar(prdGO['Product_type'], prdGO['Ordered'], color=c, width=.5)
for i, val in enumerate(prdGO['Ordered'].values):
    plt.text(i, val, float(val), horizontalalignment='center', verticalalignment='bottom', fontdict={'fontweight':500, 'size':12})

plt.gca().set_xticklabels(prdGO['Product_type'], rotation=60, horizontalalignment= 'right')
plt.title("Product type that was ordered greater times", fontsize=22)
plt.ylabel('# Orders')
plt.show()


#Yearly sales in the various states
import folium
import gmaps 
from folium import plugins
from IPython.display import HTML, display
from geopy.exc import GeocoderTimedOut
from geopy.geocoders import Nominatim
def findGeocode(city):
    try:
        geolocator = Nominatim(user_agent="your_app_name")
        return geolocator.geocode(city,timeout=None)
    except GeocoderTimedOut:
        return findGeocode(city) 
d = {}
for i in dels.State.value_counts().index:
    if findGeocode(i) != None:
            loc = findGeocode(i)
            d[i] = {'latitude':loc.latitude,'longitude':loc.longitude}
    else:
            d[i] = {'latitude':np.nan,'longitude':np.nan}

def regiongrouping(data):
    d = []
    d.append(data['Quantity'].sum())
    d.append(data['Sales'].sum())
    d.append(data['Profit'].sum())
    return pd.Series(d,index=['Orders','Sales','Profit'])

dels['Date'] = pd.to_datetime(dels['Order Date'])
dels['year'], dels['month'] = dels['Date'].dt.year, dels['Date'].dt.month
dels = dels.sort_values(['Profit'], ascending=False)
yr = sorted(list(dels['year'].unique()))
yearset = {}
for i in yr:
    longitude = []
    latitude = []
    yrs = dels[dels['year'] == i]
    regions = yrs.groupby(['State']).apply(regiongrouping)
    for j in (regions.index):
        if j in d.keys():
            longitude.append(d[j]['longitude'])
            latitude.append(d[j]['latitude'])
    regions['latitude'] = latitude
    regions['longitude'] = longitude
    tooltip = "Click Here For More Info"
    mlat = regions['latitude'].mean()
    mlong = regions['longitude'].mean()
    m = folium.Map(location=[mlat, mlong], zoom_start=5)
    folium.map.Marker(
    [mlat, mlong],
    icon=folium.DivIcon(
        icon_size=(250,36),
        icon_anchor=(0,0),
        html='<div style="font-size: 16pt; color: black">Yearly sales in the various states in {}</div>'.format(i),
        )
    ).add_to(m)
    heats = regions.Sales
    fig, ax = plt.subplots(1, figsize=(20, 9))
    for r,o,s,p,la,lo in zip(regions.index,regions['Orders'],regions['Sales'],regions['Profit'],regions['latitude'],regions['longitude']):
        c = "green" if p > 0 else "red"
        market = "Profit" if p > 0 else "Loss"
        ic ="thumbs-up" if p > 0 else "thumbs-down"
        myMap = folium.Marker(location=[la, lo],icon=folium.Icon(icon=ic,color=c), popup="<stong>{}</stong><p>Sales:{}</p><p>Orders:{}</p><p>{}:{}</p>".format(r,s,int(o),market,p),tooltip=tooltip)
        myMap.add_to(m)
    yearset[i] = m
longitude = []
latitude = []
OAregions = dels.groupby(['State']).apply(regiongrouping)
for j in (OAregions.index):
    if j in d.keys():
        longitude.append(d[j]['longitude'])
        latitude.append(d[j]['latitude'])
OAregions['latitude'] = latitude
OAregions['longitude'] = longitude
tooltip = "Click Here For More Info"
mlat = OAregions['latitude'].mean()
mlong = OAregions['longitude'].mean()
OAm = folium.Map(location=[mlat, mlong], zoom_start=5)
folium.map.Marker(
[mlat, mlong],
icon=folium.DivIcon(
    icon_size=(250,36),
    icon_anchor=(0,0),
    html='<div style="font-size: 16pt; color: black">Overall sales in the various states</div>',
    )
).add_to(OAm)
for r,o,s,p,la,lo in zip(OAregions.index,OAregions['Orders'],OAregions['Sales'],OAregions['Profit'],OAregions['latitude'],OAregions['longitude']):
    c = "green" if p > 0 else "red"
    market = "Profit" if p > 0 else "Loss"
    ic ="thumbs-up" if p > 0 else "thumbs-down"
    myMap = folium.Marker(location=[la, lo],icon=folium.Icon(icon=ic,color=c), popup="<stong>{}</stong><p>Sales:{}</p><p>Orders:{}</p><p>{}:{}</p>".format(r,s,int(o),market,p),tooltip=tooltip)
    myMap.add_to(OAm)
yearset['Overall'] = OAm
for n in yr:
    yearset[n].save("states{}.HTML".format(n))
yearset['Overall'].save("Overall_state.HTML")
