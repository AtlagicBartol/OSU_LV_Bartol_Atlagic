import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


X, y = make_classification(n_samples=200, n_features=2, n_redundant=0, n_informative=2,
                            random_state=213, n_clusters_per_class=1, class_sep=1)

X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state = 4)

# a)Prikažite podatke za ucenje u ˇ x1 −x2 ravnini matplotlib biblioteke pri cemu podatke obojite ˇ
# s obzirom na klasu. Prikažite i podatke iz skupa za testiranje, ali za njih koristite drugi
# marker (npr. ’x’). Koristite funkciju scatter koja osim podataka prima i parametre c i
# cmap kojima je moguce de ´ finirati boju svake klase.

plt.scatter(X_train[:,0], X_train[:,1], c = y_train,cmap='Pastel2',label = "Train")
plt.scatter(X_test[:,0], X_test[:,1], c = y_test,cmap='Pastel2', marker = "*",label = "Test")
plt.legend()
plt.show()

# b)Izgradite model logisticke regresije pomo ˇ cu scikit-learn biblioteke na temelju skupa poda- ´
#taka za ucenje
Regresion_model = LogisticRegression()
Regresion_model.fit(X_train,y_train)
y_test_p = Regresion_model.predict(X_test)

# c)Pronadite u atributima izgra ¯ denog modela parametre modela. Prikažite granicu odluke ¯
# naucenog modela u ravnini ˇ x1 − x2 zajedno s podacima za ucenje. Napomena: granica ˇ
# odluke u ravnini x1 −x2 definirana je kao krivulja: θ0 +θ1x1 +θ2x2 = 0.
b = Regresion_model.intercept_[0]
w1,w2 = Regresion_model.coef_.T

c = -b/w2
m = -w1/w2 

xmin, xmax = -4,4
ymin, ymax = -4,4

xd = np.array([xmin,xmax])
yd = m*xd + c

plt.plot(xd,yd, 'k', lw = 1, ls ='--')
plt.fill_between(xd, yd, ymin,color='green')
plt.fill_between(xd, yd, ymax, color='black')
plt.scatter(X_train[:,0], X_train[:,1], c =y_train,cmap='Greens',label = "Train",edgecolors='white')
plt.show()

# d)Provedite klasifikaciju skupa podataka za testiranje pomocu izgra ´ denog modela logisti ¯ cke ˇ
# regresije. Izracunajte i prikažite matricu zabune na testnim podacima. Izra ˇ cunate to ˇ cnost, ˇ
# preciznost i odziv na skupu podataka za testiranje.

cofMatrix = confusion_matrix(y_test, y_test_p)
display = ConfusionMatrixDisplay(cofMatrix)
display.plot()
plt.show()
print(classification_report(y_test, y_test_p))

# e)Prikažite skup za testiranje u ravnini x1 −x2. Zelenom bojom oznacite dobro klasi ˇ ficirane
# primjere dok pogrešno klasificirane primjere oznacite crnom bojom.
x_false = []

for i in range(len(y_test)):
    if y_test[i] != y_test_p[i]:
        x_false.append([X_test[i,0], X_test[i,1]])

x_false = np.array(x_false)
print(x_false)

plt.scatter(X_test[:,0], X_test[:,1], c="green")
plt.scatter(x_false[:,0],x_false[:,1], c = "black")
plt.show()

https://chatgpt.com/share/681c7d44-eecc-8011-8701-fc434bc8c85e

https://chatgpt.com/share/681c7ef2-6ab4-8011-9654-de3bde2ae7d9

treci al nije testiran:
https://chatgpt.com/share/681c80ce-c980-8011-9bce-2a094c2925a2
