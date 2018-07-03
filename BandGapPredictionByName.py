from pymatgen import Composition, Element
from numpy import zeros, mean

# Training file containing band gaps extracted from Materials Project
trainFile = open("bandgapDFT.csv","r").readlines()

# Input: pymatgen Composition object
# Output: length-100 vector representing any chemical formula

def naiveVectorize(composition):
       vector = zeros((MAX_Z))
       for element in composition:
               fraction = composition.get_atomic_fraction(element)
               vector[element.Z - 1] = fraction
       return(vector)

# Extract materials and band gaps into lists,construct naive feature set
materials = []
bandgaps = []
naiveFeatures = []
MAX_Z = 100 

for line in trainFile:
       split = str.split(line, ',')
       material = Composition(split[0])
       materials.append(material) #store chemical formulas
       naiveFeatures.append(naiveVectorize(material)) #create features from chemical formula
       bandgaps.append(float(split[1])) #store numerical values of band gaps
############################################
baselineError = mean(abs(mean(bandgaps) - bandgaps))
print("The MAE of always guessing the average band gap is: " + str(round(baselineError, 3)) + " eV")
# Train linear ridge regression model using naive feature set
from sklearn import linear_model, cross_validation, metrics, ensemble
linear = linear_model.Ridge(alpha = 0.5)

cv = cross_validation.ShuffleSplit(len(bandgaps),\
	n_iter=10, test_size=0.1, random_state=0)

scores = cross_validation.cross_val_score(linear, naiveFeatures,\
	bandgaps, cv=cv, scoring='mean_absolute_error')

print("The MAE of model using the naive feature set is: "\
	+ str(round(abs(mean(scores)), 3)) + " eV")

print("Below naive feature set")

linear.fit(naiveFeatures, bandgaps) # fit to the whole data set
print("element: coefficient")

for i in range(MAX_Z):
       element = Element.from_Z(i + 1)
       print(element.symbol + ': ' + str(linear.coef_[i]))
####To be continued
# more physically-motivated
physicalFeatures = []

for material in materials:
       theseFeatures = []
       fraction = []
       atomicNo = []
       eneg = []
       group = []
       for element in material:
               fraction.append(material.get_atomic_fraction(element))
               atomicNo.append(float(element.Z))
               eneg.append(element.X)
               group.append(float(element.group))

       # We want to sort this feature set
       # according to which element in the binary compound is more abundant
       mustReverse = False

       if fraction[1] > fraction[0]:
               mustReverse = True

       for features in [fraction, atomicNo, eneg, group]:
               if mustReverse:
                       features.reverse()
       theseFeatures.append(fraction[0] / fraction[1])
       theseFeatures.append(eneg[0] - eneg[1])
       theseFeatures.append(group[0])
       theseFeatures.append(group[1])
       physicalFeatures.append(theseFeatures)
