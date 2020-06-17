



# In[ ]:


import matplotlib.pylab as plt
from sklearn import linear_model

reg=linear_model.LinearRegression()

X=[[174],[152],[138],[128],[186]]
y=[71,55,46,38,88]
reg.fit(X,y) #학습

print(reg.predict([[165]]))

#학습 데이터와 y값을 산포도로 그린다.
plt.scatter(X,y,color='black')

#학습 데이터를 입력으로 하여 예측값을 계산한다.
y_pred = reg.predict(X)

#학습데이터와 예측값으로 선그래프로 그린다.
#계산된 기울기와 y절편을 가지는 직선이 그려진다.
plt.plot(X, y_pred, color='blue',linewidth=3)
plt.show()


# In[ ]:


from sklearn.datasets import load_iris  #scikit-learn의 샘플 데이터 로드
import pandas as pd
import numpy as np

#시각화를 위한 패키치 임포트
import matplotlib.pyplot as plt
import seaborn as sns

iris=load_iris()  #sample data load

print(iris)  #로드된 데이터가 속성-스타일 접근을 제공하는 딕셔너리와 번치 객체로 표현된 것을 확인
print(iris.DESCR)  #Description 속성을 이용해서 데이터 셋의 정보를 확인

#각 key에 저장된 value 확인
#feature
print(iris.data)
print(iris.feature_names)

#label
print(iris.target)
print(iris.target_names)

#feature_names 와 target을 레코드로 갖는 데이터 프레임 생성
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target']=iris.target

#0.0, 1.0, 2.0으로 표현된 label을 문자열로 매핑
df['target']=df['target'].map({0:"setosa",1:"versicolor",2:"virginica"})
print(df)

#슬라이싱을 통해 feature laber 분리
x_data=df.iloc[:, :-1]
y_data=df.iloc[:, [-1]]

sns.pairplot(df, x_vars=["sepal length (cm)"], y_vars=["sepal width (cm)"], hue="target", height=5)


# In[ ]:


#(80:20)으로 데이터 분할
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'],random_state=0)

#분류
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#예측
y_pred=knn.predict(X_test)
scores=metrics.accuracy_score(y_test,y_pred)
print(scores)


knn= KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)

#0=setosa, 1=versicolor, 2=virginica
classes={0:'setosa',1:'versicolor',2:'virginica'}

#아직 보지 못한 새로운 데이터를 제시
x_new=[[3,4,5,2],
      [5,4,2,2]]
y_predict = knn.predict(x_new)

print(classes[y_predict[0]])
print(classes[y_predict[1]])


# In[ ]:


from sklearn import datasets
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

#데이터 로드
iris=datasets.load_iris()

#입력(X)과 출력(y) or target
X=iris.data[:, :2]
y=iris.target

#데이터 살펴보기
plt.scatter(X[:,0],X[:,1],c=y,cmap='gist_rainbow')
plt.xlabel('Spea1 Length',fontsize=18)
plt.ylabel('Sepal Width',fontsize=18)

#k-means 크러스터링
km= KMeans(n_clusters=3,n_jobs=4,random_state=21)
km.fit(X)

#중심점 위치
centers=km.cluster_centers_
print(centers)


# In[ ]:


import os
import pandas as pd
import matplotlib.pyplot as plt

os.chdir('C:/Users/박시윤')
iris= pd.read_csv('iris.csv')

print(iris.head())

iris.loc[iris['species']=='virginica','species']=0
iris.loc[iris['species']=='versicolor','species']=1
iris.loc[iris['species']=='setosa','species']=2
iris=iris[iris['species']!=2]

X=iris[['sepal_length','sepal_width']].values.T
Y=iris[['species']].values.T
Y=Y.astype('uint8')

plt.scatter(X[0,:],X[1,:],c=Y[0,:],s=40,cmap=plt.cm.Spectral)
plt.title("IRIS DATA Blue - Versicolor, Red-Virginica")
plt.xlabel('petal_length')
plt.ylabel('petal_width')
plt.show()


# In[ ]:


import seaborn as sns
import pandas as pd
import numpy as np

sns.set(style="ticks",color_codes=True)
iris=sns.load_dataset("iris")
g=sns.pairplot(iris, hue="species", palette="husl")


# In[ ]:


iris.info()
iris['species'].unique()


# In[ ]:


from sklearn.preprocessing import LabelEncoder
X=iris.iloc[:,0:4].values
y=iris.iloc[:,4].values

encoder=LabelEncoder()
y1=encoder.fit_transform(y)
Y=pd.get_dummies(y1).values

print(Y)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                   test_size=0.2,
                                                   random_state=1)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


# In[ ]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model=Sequential()

model.add(Dense(64,input_shape=(4,),activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(3,activation='softmax'))

model.compile(loss='categorical_crossentropy',
             optimizer='Adam',
             metrics=['accuracy'])

model.summary()


# In[ ]:


hist=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=100)


# In[ ]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.legend(['loss','val_loss','accuracy','val_accuracy'])
plt.grid()
plt.show()


# In[ ]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np
np.random.seed(7)

print('Python version:',sys.version)
print('Tensorflow version:',tf.__version__)
print('Keras version:',keras.__version__)


# In[ ]:


mnist=keras.datasets.mnist
(X_train0,y_train0),(X_test0,y_test0)=mnist.load_data()

import matplotlib.pylab as plt

plt.figure(figsize=(6,1))
for i in range(36):
    plt.subplot(3,12,i+1)
    plt.imshow(X_train0[i],cmap="gray")
    plt.axis("off")
plt.show()


# In[ ]:


img_rows=28
img_cols=28
(x_train,y_train),(x_test,y_test)=keras.datasets.mnist.load_data()

input_shape=(img_rows, img_cols, 1)
x_train=x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test=x_test.reshape(x_test.shape[0],img_rows,img_cols,1)

x_train=x_train.astype('float32')/255.
x_test=x_test.astype('float32')/255.

print('x_train shape:',x_train.shape)
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test samples')

batch_size =128
num_classes=10
epochs=12

y_train=keras.utils.to_categorical(y_train, num_classes)
y_test=keras.utils.to_categorical(y_test, num_classes)


# In[ ]:


model=Sequential()
model.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),padding='same',
                 activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(64,(2,2), activation='relu',padding='same'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


# In[ ]:


model.compile(loss='categorical_crossentropy',
             optimizer='adam',metrics=['accuracy'])

hist=model.fit(x_train,y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test,y_test))


# In[ ]:


score=model.evaluate(x_test,y_test,verbose=0)
print('Test loss:',score[0])
print('Test accuracy:',score[1])

import numpy as np
y_vloss=hist.history['val_loss']
y_loss=hist.history['loss']
x_len=np.arange(len(y_loss))
plt.plot(x_len,y_vloss,marker='.',c="red",label='Testset_loss')
plt.plot(x_len,y_loss,marker='.',c="blue",label='Trainset_loss')
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


# In[ ]:


n=0
plt.imshow(x_test[n].reshape(28,28), cmap='Greys',interpolation='nearest')
plt.show()
print('The Answer is ',model.predict_classes(x_test[n].reshape((1,28,28,1))))


