#!/usr/bin/env python
#
# Copyright (C) 2018  Maha Farhat
#
# Authors: Michael Chen, Martin Owens
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the 
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""
Python script and sample data from prediction from genotypic data (that can be obtained from the vcf) to a prediction in probabilities that can be displayed on the predict page. To replace TBpredict.R portion of the pipeline.

This script requires Keras 1.2.0 and TensorFlow 0.10.0
"""

import os
import sys

import numpy as np
import pandas as pd

import keras.backend as K
from keras.layers.convolutional import *
from keras.models import model_from_json

from argparse import ArgumentParser, ArgumentTypeError

def filename(name):
    filename = os.path.expanduser(name)
    filename = os.path.abspath(filename)
    if not os.path.isfile(filename):
        raise ArgumentTypeError("File not found: %s" % name)
    return filename

if __name__ == '__main__':
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Multitask WDNN Model', type=filename, default='multitask_wdnn_021318.json')
    parser.add_argument('weights', help='Multitask WDNN Weights', type=filename, default='multitask_wdnn_weights_021318.h5')
    parser.add_argument('genotype_data', help='Genotype data csv', type=filename)
    parser.add_argument('phenotype_data', help='Phenotype data csv', type=filename)
    arg = parser.parse_args()

    # To import model and weights
    with open(arg.model, 'r') as fhl:
        loaded_model = model_from_json(fhl.read())

    loaded_model.load_weights(arg.weights)

    # Get test data
    df_X_test = pd.read_csv(arg.genotype_data)
    df_y_test = pd.read_csv(arg.phenotype_data)

    X_test = df_X_test.as_matrix()
    y_test = df_y_test.as_matrix()

    # Ensembling
    def ensemble(X, y, function):
        preds = np.zeros_like(y, dtype=np.float)
        for i in range(100):
            preds += np.squeeze(np.array(function([X, 1])), axis=0)
        return preds / 100

    clf_dom = K.Function(loaded_model.inputs + [K.learning_phase()], loaded_model.outputs)

    ## NOTE: A PREDICTION OF 0 CORRESPONDS TO A "RESISTANT" PHENOTYPE AND
    ## A PREDICTION OF 1 CORREPONDS TO A "SUSCEPTIBLE" PHENOTYPE.
    y_pred = ensemble(X_test, y_test, clf_dom)

    np.savetxt(sys.stdout, y_pred, delimiter=",",
               header="rif, inh, pza, emb, str, cip, cap, amk, moxi, oflx, kan")

