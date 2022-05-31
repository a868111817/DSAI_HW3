import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # load training data
    df = pd.read_csv('training_data/target0.csv')

    # transfer column-time to datetime type
    df['time'] = pd.to_datetime(df['time'], format="%Y-%m-%d %H:%M:%S")

    df_test = df.loc[df['time'].between('2018-01-01','2018-01-31 23:00:00')]

    date = df_test.time
    generation = df_test.generation

    plt.plot(date,generation)
    plt.savefig('img/generation.png')