"""
Train DPK models for stations around the globe, 
so that they can be used in multidimensional anomaly detecton in anomaly-detection-regions.ipynb
"""

import random
import torch
import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
from datetime import datetime
import os
import time
import pickle
import json

seed = 633

print("[ Using Seed : ", seed, " ]")

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from dpk.koopman_probabilistic import KoopmanProb
from dpk.model_objs import NormalNLL


def train(station_fname, debug=False):
    start_date = np.datetime64("2018-01-01")  # sometimes there's data from before jan 1 2018, let's ignore that
    end_date = np.datetime64("2021-01-01") # exclusive
    t_min = time.mktime(dt.datetime(2018, 1, 1).timetuple())

    station_dir = "./all/"
    if station_fname.startswith("obs"):
        pathname = os.path.join(station_dir, station_fname)
        df = pd.read_csv(pathname, parse_dates=['ISO8601',], date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')).sort_values(by=['ISO8601'])
        obs = df[(df.ISO8601 >= start_date) & (df.ISO8601 < end_date) & (df.conc_obs > 0) & (df.obstype=='no2')].drop_duplicates(subset="ISO8601")
        obs["conc_obs"] = np.log(obs.conc_obs)
        obs["t"] = [time.mktime(obs.ISO8601.iloc[i].timetuple()) - t_min for i in range(len(obs))]
        
    if station_fname.startswith("model_"):
        pathname = os.path.join(station_dir, station_fname)
        df = pd.read_csv(pathname, parse_dates=['ISO8601',], date_parser=lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ')).sort_values(by=['ISO8601'])
        # df.ISO8601 -= dt.timedelta(minutes=30)  # the timestamps in mod were misaligned
        mod = df[(df.ISO8601 >= start_date) & (df.ISO8601 < end_date) & (df.NO2 > 0)].drop_duplicates(subset="ISO8601")
        mod["NO2"] = np.log(mod.NO2)
        mod.index = mod.ISO8601
        mod["t"] = [time.mktime(mod.ISO8601.iloc[i].timetuple()) - t_min for i in range(len(mod))]
    
    # set up the time-series for prediction
    if station_fname.startswith("obs"):
        x = np.expand_dims(obs.conc_obs.values, -1)
        t = obs.t.values
    else:
        x = np.expand_dims(mod.NO2.values, -1)
        t = mod.t.values
    print("x", x.shape)
    print("t", t.shape)
    null_idx = np.isnan(x).nonzero()[0]
    print(null_idx)
    x[null_idx] = (x[null_idx - 1] + x[null_idx + 1]) / 2

    scale = np.std(x, axis=0)
    x = x / np.tile(scale, (x.shape[0], 1))
    if debug:
        print("x", x.shape)
        print("t", t.shape)
        # plt.figure()
        # plt.plot(t, x)
    # split time-series
    train_start_date = np.datetime64("2018-01-01")
    train_end_date = np.datetime64("2020-01-01")  # covid start date
    covid_end_date = np.datetime64("2021-01-01")
    if station_fname.startswith("obs"):
        train_start = np.argwhere(obs.ISO8601.values >= train_start_date)[0, 0]
        train_end = np.argwhere(obs.ISO8601.values <= train_end_date)[-1, 0]
        covid_end = np.argwhere(obs.ISO8601.values <= covid_end_date)[-1, 0]
    else:
        train_start = np.argwhere(mod.ISO8601.values >= train_start_date)[0, 0]
        train_end = np.argwhere(mod.ISO8601.values <= train_end_date)[-1, 0]
        covid_end = np.argwhere(mod.ISO8601.values <= covid_end_date)[-1, 0]

    x = x[:covid_end]
    t = t[:covid_end]

    x_train = x[train_start:train_end]
    t_train = t[train_start:train_end]

    now = ("_".join(str(datetime.now()).split())).replace(":", ".")
    print("x", x.shape)
    print("x_train", x_train.shape)
    # plt.figure()
    # plt.plot(t_train, x_train)
    # plt.show()
    train_start, train_end, covid_end
    # model hyperparameters
    periods = 60 * 60 * np.array([24, 24 * 7, 24 * 365.24], dtype=np.float64)  # seconds
    l1width = 256
    l2width = 1024
    wd = 1e-3
    lrt = 1e-4
    batch_size = 32
    model = NormalNLL(x_dim=1, num_freqs=3, n=l1width, n2=l2width, num_covariates=1)  # The covariate is time

    # load a pre-trained DPK model
    total_iters = 200  # this indicates how long the pre-trained DPK model was trained
    param_str = f"NormalNLL_{station_fname[:-4]}_{l1width}_{l2width}_{batch_size}_{total_iters}_{seed}_{wd}_{lrt}"  # you must still run 1 iteration of training on this model just to initialize the koop
    if os.path.exists(f"forecasts/model_{param_str}.pt") \
        and os.path.exists(f"forecasts/params_{param_str}.npy") \
        and os.path.exists(f"forecasts/x_{param_str}.npy") \
        and os.path.exists(f"forecasts/t_{param_str}.npy") \
        and os.path.exists(f"forecasts/koop_{param_str}.pkl"):
        print("\n\nALREADY COMPLETED", station_fname)
        return
    iters = 200  # you must run at least 1 iteration in order for the time covariate to be scaled correctly
    print(f"Outputs for \"{param_str}\" does not exist. Training new model for {iters} iters.")
    param_str = f"NormalNLL_default_{l1width}_{l2width}"  # use this pre-trained model to initialize a new model to be trained
    model.load_state_dict(torch.load(f"forecasts/model_{param_str}.pt"))

    koop = KoopmanProb(model, batch_size=batch_size, device="cpu")  # this koop object does the training and prediction for you
    koop.init_periods(periods)
    total_iters = 0
    total_iters += iters
    koop.fit(x_train, t_train, covariates=t_train.reshape(len(t_train), 1), iterations=iters, cutoff=0, weight_decay=wd, lr_theta=lrt, lr_omega=0, verbose=False)

    # predict the mean and standard deviation
    params = koop.predict(t, covariates=t.reshape(len(t), 1))
    
    param_str = f"NormalNLL_{station_fname[:-4]}_{l1width}_{l2width}_{batch_size}_{total_iters}_{seed}_{wd}_{lrt}"
    np.save(f"forecasts/params_{param_str}.npy", np.array(params))
    np.save(f"forecasts/x_{param_str}.npy", x)
    np.save(f"forecasts/t_{param_str}.npy", t)
    torch.save(model.state_dict(), f"forecasts/model_{param_str}.pt")
    pickle.dump(koop, open(f"forecasts/koop_{param_str}.pkl", 'wb'))  # koop = pickle.load(open('f"forecasts/koop_{param_str}.pkl", 'rb'))


def main():
    with open("data/metadata.json") as f:
        metadata = json.loads(f.read())
    quality_thresh = 0.5

    cluster_ids = {"seattle"}
    temp = []
    for fname in metadata:
        temp.append(metadata[fname])
        temp[-1]["fname"] = fname
    df_meta = pd.DataFrame(temp)
    rows = df_meta.loc[[(bool(cluster_ids.intersection(r.clusters)) and r.quality > quality_thresh) for r in df_meta.iloc]]
    for row in rows.iloc:
        station_fname = row.fname
        print("\n"*2)
        print("WORKING ON", station_fname)
        train(station_fname, debug=True)

if __name__ == "__main__":
    main()
