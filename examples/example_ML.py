import sys 
from sklearn.linear_model import ElasticNet
from sklearn import metrics, model_selection
import xgboost
import argparse
import os
os.chdir(os.path.join(os.pardir,'pyneuromodulation'))
sys.path.append(os.path.join(os.pardir,'pyneuromodulation'))

import nm_decode
from skopt.space import Real, Integer, Categorical

if __name__ == "__main__":

    PATH_FEATURES = os.path.join(os.getcwd(),r"tests\data\derivatives" )
    FEATURE_FILE = r"sub-testsub_ses-EphysMedOff_task-buttonpress_ieeg"

    parser = argparse.ArgumentParser()
    parser.add_argument('-f','--file',type=str,required=False)
    parser.add_argument('-m', '--model',type=str,required=False)

    args = parser.parse_args()

    if args.file != None:
        FEATURE_FILE =args.file


    model = xgboost.XGBRegressor()

    if args.model == "xgboost":
        model = xgboost.XGBRegressor()

    if args.model == "elasticnet":
        model = ElasticNet(max_iter=10000)


    decoder = nm_decode.Decoder(feature_path=PATH_FEATURES,
                                feature_file=FEATURE_FILE,
                                model=model,
                                eval_method=metrics.mean_absolute_error,
                                cv_method=model_selection.KFold(n_splits=3, shuffle=False),
                                threshold_score=True
                                )
    # estimate model performance directly
    # individual results will then be stored in the decoder object

    test_score_mean = decoder.run_CV()

    # run bayesian optimization
    # special attention needs to be made with the run_CV output, some metrics are minimized (MAE), some are maximized (r^2)
    space_XGB = [Integer(1, 100, name='max_depth'),
                 Real(10**-5, 10**0, "log-uniform", name='learning_rate'),
                 Real(10**0, 10**1, "uniform", name="gamma")]

    # other space for ElasticNet
    space_ENet = [Real(0, 1, "uniform", name='alpha'),
                  Real(0, 1, "uniform", name='l1_ratio')]

    base_estimator = "GP"
    acq_func = "EI"
    acq_optimizer = "sampling"
    initial_point_generator = "lhs"
    res_skopt = decoder.run_Bay_Opt(space_XGB, rounds=10, base_estimator=base_estimator,
                                    acq_func=acq_func, acq_optimizer=acq_optimizer,
                                    initial_point_generator=initial_point_generator)
    best_metric = res_skopt.fun
    best_params = res_skopt.x

    decoder.train_final_model(bayes_opt=True)

    decoder.save()
