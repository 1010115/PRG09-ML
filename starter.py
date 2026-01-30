"""
Startercode bij Lesbrief: Machine Learning, CMTPRG01-9

Deze code is geschreven in Python3

Benodigde libraries:
- NumPy
- SciPy
- matplotlib
- sklearn

"""
from machinelearningdata import Machine_Learning_Data
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression


def extract_from_json_as_np_array(key, json_data):
    """ helper functie om data uit de json te halen en om te zetten naar numpy array voor sklearn"""
    data_as_array = []
    for p in json_data:
        data_as_array.append(p[key])
    
    return np.array(data_as_array)


STUDENTNUMMER = "1010115"

assert STUDENTNUMMER != "1234567", "Verander 1234567 in je eigen studentnummer"

print("STARTER CODE")

# maak een data-object aan om jouw data van de server op te halen
data = Machine_Learning_Data(STUDENTNUMMER)

# UNSUPERVISED LEARNING

# haal clustering data op
kmeans_training = data.clustering_training()

# extract de x waarden
X = extract_from_json_as_np_array("x", kmeans_training)

#print(X)

# slice kolommen voor plotten
x = X[:, 0]
y = X[:, 1]

# teken de punten
plt.plot(x,y, 'k.') 
plt.axis([min(x), max(x), min(y), max(y)])

#plt.show()

# TODO: print deze punten uit en omcirkel de mogelijke clusters
guess = np.array([[12,9],[60,65],[80,9],[83,30],[95,65]])

ax = plt.gca()
for center in guess:
    circle = Circle(center, radius=20, ec="blue", fill=False, linewidth=2)
    ax.add_patch(circle)

#plt.show()

# TODO: ontdek de clusters mbv kmeans en teken een plot met kleurtjes
plt.figure(figsize=(15, 5))

for i, k_value in enumerate([5, 6, 4],1):
    plt.subplot(1, 3, i)
    kmeans = KMeans(n_clusters=k_value).fit(X)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    plt.scatter(x, y, c=labels, alpha=0.5)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
    plt.title(f"K-Means (k={k_value})")


#plt.show()

# SUPERVISED LEARNING

# haal data op voor classificatie
classification_training = data.classification_training()


# extract de data x = array met waarden, y = classificatie 0 of 1
X = extract_from_json_as_np_array("x", classification_training)

# dit zijn de werkelijke waarden, daarom kan je die gebruiken om te trainen
Y = extract_from_json_as_np_array("y", classification_training)



# TODO: leer de classificaties
x= X[:,0]
y= X[:,1]

plt.figure()
plt.scatter(x,y, c=Y) 

# plt.show()
lnRegression = LogisticRegression().fit(X,Y)

# Decision boundary
coef = lnRegression.coef_[0]
intercept = lnRegression.intercept_[0]
line_x = np.linspace(min(x),max(x))
line_y = -(coef[0] * line_x + intercept) / coef[1]

plt.plot(line_x, line_y, color='red')


# TODO: voorspel na het trainen de Y-waarden (je gebruikt hiervoor dus niet meer de
#       echte Y-waarden, maar enkel de X en je getrainde classifier) en noem deze
#       bijvoordeeld Y_predict
plt.figure()
Y_predict = lnRegression.predict(X)
plt.scatter(x,y, c=Y_predict) 



# TODO: vergelijk Y_predict met de echte Y om te zien hoe goed je getraind hebt
print("Training accuracy "+ str(accuracy_score(Y,Y_predict)))
# plt.show()

# haal data op om te testen
classification_test = data.classification_test()
# testen doen we 'geblinddoekt' je krijgt nu de Y's niet
X_test = extract_from_json_as_np_array("x", classification_test)
# TODO: voorspel na nog een keer de Y-waarden, en plaats die in de variabele Z
#       je kunt nu zelf niet testen hoe goed je het gedaan hebt omdat je nu
#       geen echte Y-waarden gekregen hebt.
#       onderstaande code stuurt je voorspelling naar de server, die je dan
#       vertelt hoeveel er goed waren.
X_predict = lnRegression.predict(X_test)
plt.figure()
plt.scatter(X_test[:,0],X_test[:,1], c=X_predict) 
plt.plot(line_x, line_y, color='red')


Z = X_predict

# stuur je voorspelling naar de server om te kijken hoe goed je het gedaan hebt
classification_test = data.classification_test(Z.tolist()) # tolist zorgt ervoor dat het numpy object uit de predict omgezet wordt naar een 'normale' lijst van 1'en en 0'en
print("Classificatie accuratie (test): " + str(classification_test))

plt.show()