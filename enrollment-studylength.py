import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

df = pd.read_csv('SearchResults.csv')
#df['Study length'] = df['Completion Date']-df['Start Date']
df['Study length'] = (pd.to_datetime(df['Completion Date'])-pd.to_datetime(df['Start Date']))
df = df[['Study Designs', 'Enrollment', 'Study length']]
df[['Allocation', 'Intervention Model', 'Masking', 'Primary Purpose']] = df['Study Designs'].str.split('|', 3, expand=True)
df = df.drop(columns=['Study Designs', 'Intervention Model', 'Masking'])
df['Allocation'] = df['Allocation'].str.replace('Allocation: ', "")
df['Primary Purpose'] = df['Primary Purpose'].str.replace('Primary Purpose: ', "")
#df['Study length'] = (df['Study length']).replace(' days',"")
df['Study length'] = df['Study length'].dt.days


#Dummy variables: 0- Non-Randomized or N/A, 1- Randomized
df.loc[df['Allocation']!='Randomized', 'Allocation']=0
df.loc[df['Allocation']=='Randomized', 'Allocation']=1
#print(df.dtypes)


df1 = df.loc[df['Primary Purpose']=='Treatment']
df1=df1.drop(columns='Primary Purpose')
#print(df1)
df1 = df1.drop(columns='Allocation')

X = df1.drop(columns=['Study length'])
y = df1['Study length']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
#print("train")
print(X_train)
#print("test")
print(X_test)

plt.scatter(df1['Enrollment'], df1['Study length'])
plt.show()


scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

sgdr = SGDRegressor(max_iter=1000)
sgdr.fit(X_norm, y_train)
print(sgdr)
print(f"number of iterations completed: {sgdr.n_iter_}, number of weight updates: {sgdr.t_}")

b_norm = sgdr.intercept_
w_norm = sgdr.coef_
print(f"model parameters:                   w: {w_norm}, b:{b_norm}")



# make a prediction using sgdr.predict()
y_pred_sgd = sgdr.predict(X_norm)
# make a prediction using w,b. 
y_pred = np.dot(X_norm, w_norm) + b_norm  
print(f"prediction using np.dot() and sgdr.predict match: {(y_pred == y_pred_sgd).all()}")

print(f"Prediction on training set:\n{y_pred[:4]}" )
print(f"Target values \n{y_train[:4]}")

'''
x = np.linspace(-5,5,100)
y_line = 2*x+1
plt.plot(x,y_line)
plt.show()
'''


'''
ln_Y = np.log(y)
exp_reg = LinearRegression()
exp_reg.fit(X, ln_Y)
#### You can introduce weights as well to apply more bias to the smaller X values, 
#### I am transforming X arbitrarily to apply higher arbitrary weights to smaller X values
exp_reg_weighted = LinearRegression()
#exp_reg_weighted.fit(X, ln_Y, sample_weight=np.array(1/((X - 100).values**2)).reshape(-1))
exp_reg_weighted.fit(X, ln_Y, sample_weight=np.array(1/((X - 100)**2)).reshape(-1))

### Get predicted values of Y
Y_pred = np.exp(exp_reg.predict(X))
Y_pred_weighted = np.exp(exp_reg_weighted.predict(X))

plt.scatter(X, y)
plt.plot(X, Y_pred, label='Default')
plt.plot(X, Y_pred_weighted, label='Weighted')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.show()
'''






