
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import utils
import pickle
import datetime

def strategy(gen_predict,con_predict,df):
    action = 'no action' # -1 -> sell ,0 -> no action ,1 -> buy
    volume = 0.0
    price = 0.0

    if (gen_predict > con_predict):
        action = 'buy'
        volume = gen_predict - con_predict
        price = 2.6
    elif (gen_predict == con_predict):
        action = 'no action'
    else:
        action = 'sell'
        volume = con_predict - gen_predict
        price = 2.4

    tomorrow = df['time'].iloc[6] + datetime.timedelta(days=1)
    data = [[tomorrow,action,price,volume]]

    return data

def config():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()

    # load generation.csv
    gen = pd.read_csv(args.generation)

    # load consumption.csv
    con = pd.read_csv(args.consumption)

    # transfer column-time to datetime type
    gen['time'] = pd.to_datetime(gen['time'], format="%Y-%m-%d %H:%M:%S")
    con['time'] = pd.to_datetime(con['time'], format="%Y-%m-%d %H:%M:%S")

    # merge two dataframe
    df = pd.merge(gen,con)

    # transfer testing data from hours to days
    df = utils.day_generation(df,7)

    # load model
    model = load_model('rnn_model.h5')
    # save scale
    scalerfile = 'sc.sav'
    sc = pickle.load(open(scalerfile, 'rb'))

    # transfer testing datatype
    gen_test, con_test = utils.iterator_test(df,sc)

    # predict generation
    prediction = model.predict(gen_test)
    prediction = sc.inverse_transform(prediction)
    gen_predict = float(prediction)

    # predict consumption
    prediction = model.predict(con_test)
    prediction = sc.inverse_transform(prediction)
    con_predict = float(prediction)

    data = strategy(gen_predict,con_predict,df)

    output(args.output, data)
