from helpers import *
from keras.layers import Dense, Dropout, Input, merge
from keras.models import Model
from keras.optimizers import Adam

X = np.loadtxt("X_features.csv", delimiter=',')
alpha_matrix = np.loadtxt("alpha_matrix.csv", delimiter=',')

# Train multitask WDNN on entire data set
clf = get_wide_deep()
clf.fit(X, alpha_matrix, nb_epoch=100)

# Get final layer of model
#dom = K.Function(clf.inputs + [K.learning_phase()], clf.outputs)
embedding = clf.predict(X)

# Save weights
np.savetxt('embedding.csv', embedding, delimiter=',')
