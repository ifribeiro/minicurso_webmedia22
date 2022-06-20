from cProfile import label
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_list_dates(data_size=None, year=None, month=None, day=1, tdeltah=1):
    """
    Creates a list of dates for the dataset. It depends on the data size and data frequency
    Params:
        
        - data_size: number of samples in data
        - y: year
        - m: month
        - d: day
    """
    date = datetime.datetime(year,month,day,0)
    tdelta = datetime.timedelta(hours=tdeltah)
    list_dates = []
    for i in range(data_size):
        list_dates.append(date)
        date = date + tdelta    
    return list_dates

def get_count(df, wkday, timestep,column='hr'):
    """
    Sums the variables in the data by each interval
    TODO: accetp weekday as list of days
    Params:

        - df: A dataset with the date, hour, week, minutes and variables columns
        - wkday: weekday to filter the data
        - timestep: the data timestep
        - column: column in the data that will be used for computing the sums
    """
    df_wkday = df[df['wk'] == wkday]
    dict_cnt_column = dict()       
    for col in df.columns:
        if "cnt" in col:
            dict_cnt_column[col] = [df_wkday[df_wkday[column]==i][col].sum() for i in range(timestep)]
    
    return dict_cnt_column

def add_week_hr(df, timesteps=24):
    """
    TODO: test when hour = 2

    Add additional columns (hour, week and minutes) to the dataframe

    Params

        - df: a pandas dataframe. Must have a column named 'date' that is a datetime object
    """
    df['wk'] = df.apply(lambda r: r.date.weekday(), axis=1)

    return df

def get_df(list_dates=None, data=None, timesteps=24):
    """
    Returns a pandas dataframe the columns to be used in the visualizations

    Params:

        - list_dates: list of dates
        - real_data: the real data that will be used to create the pandas dataframe
    """
    
    dict_real = {'date':list_dates}
    for i in range(data.shape[1]):
        dict_real['cnt_{}'.format(i)] = data[:,i]

    df_real = pd.DataFrame(dict_real)
    df_real = add_week_hr(df_real)
    ts = [t for t in range(timesteps)]
    df_real['ts'] = ts * (data.shape[0]//timesteps)
    return df_real

def plot_sum_real(list_cnt_real = [], fake_data = None, list_dates=None, figtitle="",
                    wkday=0, plots_dir="", timestep=0, fmt="png"):
    """

    """
    wks_name=['Seg.', 'Ter.', 'Qua.', 'Qui.', 'Sex.', 'Sab.', 'Dom.']
    markers = ['.', '1', '*', '+', 'o', '^', 'v','']
    fig, ax = plt.subplots(figsize=(8,5))
    for i, cnt_real in enumerate(list_cnt_real):
        ax.plot(cnt_real,label='{}'.format(wks_name[i]), marker=markers[i])
    ax.set_xlabel("Hora", fontdict={'fontsize':14}) 
    ax.set_ylabel("Bicicletas alugadas", fontdict={'fontsize':14})
    plt.title("{}".format(figtitle))
    plt.legend()
    plt.show()
    plt.clf()
    plt.close()

def get_list_wks(model_samples, real_data, list_dates_real, timestep):
    """    
    params:

    - real data (for shape)
    - list dates
    - wk
    """
    list_wks = []
    for i, s in enumerate(model_samples):
        # reshape
        s = np.load(s)
        s = s.reshape(len(list_dates_real), 1)
        # s = s[:3287]
        # s = np.exp(s)
        # s = s[:, :, idx]
        #sample_reshaped = s.reshape(250, 48, 1)
        df_sample = get_df(list_dates_real, s, timesteps=timestep)    
        wk_fake = get_count(df_sample,4,timestep=timestep, column='ts')
        list_wks.append(wk_fake['cnt_0'])
        if (i % 2 == 0):
            print ("Done {}!".format(i))
    return list_wks

def plot_compare_sum(real, somas, std):
    fig, ax = plt.subplots()
    x = np.arange(0,24,1)
    ax.plot(x, real, label='real', marker='.',ls='--')
    ax.plot(x, somas)
    ax.fill_between(x, somas-std, somas+std, alpha=0.3)
    ax.set_xlabel("Hora",fontdict={'fontsize':14})
    ax.set_ylabel("Bicicletas alugadas",fontdict={'fontsize':14})
    plt.legend()