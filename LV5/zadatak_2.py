import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay



def plot_decision_regions(X,y,classifier,resolution = 0.02):
    plt.figure()
    markers = ('s', 'x', 'o')
    colors = ('red', 'blue', 'lightgreen')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:,0].min() - 1, X[:,0].max() + 1
    x2_min,x2_max = X[:,1].min() - 1, X[:,1].max() + 1
    x1,x2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
    np.arange(x2_min,x2_max,resolution))
    Z = classifier.predict(np.array([x1.ravel(),x2.ravel()]).T)
    Z = Z.reshape(x1.shape)
    plt.contourf(x1,x2,Z,alpha=0.5,cmap = cmap)
    plt.xlim(x1.min(), x1.max())
    plt.ylim(x2.min(), x2.max())

    for idx,c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y==c1,0],
                    y=X[y==c1,1],
                    alpha = 0.8,
                    c=colors[idx],
                    marker = markers[idx],
                    edgecolors ='w',
                    label = labels[c1])
    plt.show()


labels= {0:'Adelie', 1:'Chinstrap', 2:'Gentoo'}

data = pd.read_csv('penguins.csv')

data.drop_duplicates()

data['species'].replace({'Adelie' : 0,
                         'Chinstrap' : 1,
                         'Gentoo' : 2}, inplace = True)

print(data.info())

data.dropna(axis=0,inplace=True)

output= ['species']

input= ['bill_length_mm', 'flipper_length_mm']

x = data[input].to_numpy()
y = data[output].to_numpy()[:,0]

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=1)

# classes, count_train = np.unique(y_train, return_counts = True)
# classes, counts_test = np.unique(y_test,return_counts = True)

# plt.bar(classes, count_train, 0.4, label = 'Train')
# plt.bar(classes + 0.2, counts_test, 0.4, label = 'Test')
# plt.xticks(classes,['Adelie(0)', 'Chinstrap(1)', 'Gentoo(2)'])
# plt.xlabel("Penguins")
# plt.ylabel("Counts")
# plt.title("Number of each class of penguins, train and test data")
# plt.legend()
# plt.show()

# b)
model = LogisticRegression(max_iter = 120, multi_class='ovr', solver = 'newton-cg')


model.fit(X_train,y_train)

theta0 = model.intercept_
coefs = model.coef_

print('Theta0:')
print(theta0)
print('Parametri modela') 
print(coefs) 

plot_decision_regions(X_train,y_train,model)

y_p = model.predict(X_test)
confMatrix = confusion_matrix(y_test,y_p)
display = ConfusionMatrixDisplay(confMatrix)
display.plot(cmap='cividis')
plt.title('Confusion Matrix')
plt.show()

print(f"Tocnost: {accuracy_score(y_test,y_p)}")
print(classification_report(y_test,y_p))
