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

def count_daily(df_real):
    groupsMonth = df_real.groupby(by="month") 
    meses = np.arange(1,13)
    dict_meses_day = {}
    dict_meses_wk  = {}
    for m in meses:
        dict_meses_day.setdefault(m,{"x":[],"cnts":[]})
        dict_meses_wk.setdefault(m, {"x":[],"cnts":[]})
        mes = groupsMonth.get_group(m)
        groupsDay = mes.groupby(by="day")
        groupsWk  = mes.groupby(by="wk")
        x_valores_day = list(groupsDay.groups.keys())
        x_valores_wk  = list(groupsWk.groups.keys())
        lista_cnt = []
        list_cnt_wk = []
        for k in x_valores_day:
            df = groupsDay.get_group(k)
            cnt = df['cnt_0'].mean()
            lista_cnt.append(cnt)
        
        for k in x_valores_wk:
            df = groupsWk.get_group(k)
            cnt = df['cnt_0'].mean()
            list_cnt_wk.append(cnt)

        dict_meses_day[m]["x"] = x_valores_day
        dict_meses_day[m]["cnts"] = lista_cnt

        dict_meses_wk[m]["x"] = x_valores_wk
        dict_meses_wk[m]["cnts"] = list_cnt_wk
    return dict_meses_day, dict_meses_wk        

def add_week_hr(df, timesteps=24, month=False):
    """
    Add additional columns (hour, week and minutes) to the dataframe

    Params

        - df: a pandas dataframe. Must have a column named 'date' that is a datetime object
    """
    df['wk'] = df.apply(lambda r: r.date.weekday(), axis=1)
    if month:
        df['month'] = df.apply(lambda r: r.date.month, axis=1)
        df['day'] = df.apply(lambda r: r.date.day, axis=1)

    return df

def get_df(list_dates=None, data=None, timesteps=24, month=False):
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
    df_real = add_week_hr(df_real, month=month)
    if not month:
        ts = [t for t in range(timesteps)]
        df_real['ts'] = ts * (data.shape[0]//timesteps)
    return df_real

def plot_sum_real(list_cnt_real = [], fake_data = None, list_dates=None, figtitle="",
                    wkday=0, plots_dir="", timestep=0, fmt="png"):
    """

    """
    wks_name=['Seg.', 'Ter.', 'Qua.', 'Qui.', 'Sex.', 'Sab.', 'Dom.']
    markers = ['.', '1', '*', '+', 'o', '^', 'v','']
    fig, ax = plt.subplots(figsize=(6,4))
    for i, cnt_real in enumerate(list_cnt_real):
        ax.plot(cnt_real,label='{}'.format(wks_name[i]), marker=markers[i])
    ax.set_xlabel("Hora", fontdict={'fontsize':14}) 
    ax.set_ylabel("Bicicletas alugadas", fontdict={'fontsize':14})
    plt.title("{}".format(figtitle))
    plt.legend()

def get_list_wks(model_samples, list_dates_real, timestep, wk=0):
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
        s = s.flatten()[:len(list_dates_real)]
        s = s.reshape(len(list_dates_real), 1)
        df_sample = get_df(list_dates_real, s, timesteps=timestep)    
        wk_fake = get_count(df_sample,wk,timestep=timestep, column='ts')
        list_wks.append(wk_fake['cnt_0'])
    return list_wks

def plot_compare_sum(dict_weeks, dict_weeks_real, bbox, figtitle="", scaler=None):

    fig = plt.figure(figsize=(15,8))
    # top plots
    top_1 = fig.add_subplot(3, 3, (1, 1))
    top_2 = fig.add_subplot(3, 3, (2, 2))
    top_3 = fig.add_subplot(3, 3, (3, 3))
    # mid plots
    mid_1 = fig.add_subplot(3, 3, (4, 4))
    mid_2 = fig.add_subplot(3, 3, (5, 5))
    mid_3 = fig.add_subplot(3, 3, (6, 6))
    # bottom plot
    bottom = fig.add_subplot(3, 3, (8, 8))
    axes = [top_1, top_2, top_3, mid_1, mid_2, mid_3, bottom]    
    
    x = np.arange(0, 24, 1) 
    wks_name=['Real', 'Seg.', 'Ter.', 'Qua.', 'Qui.', 'Sex.', 'Sab.', 'Dom.']
    colors = ['orange','coral', 'fuchsia','purple', 'red','green','brown']
    # para customizar a legenda
    lista_handles = []
    lista_ylabel = [0, 3, 6]
    lista_xlabel = [3, 5, 6]
    for wk in range(7):
        a = np.array(dict_weeks[wk])
        real = np.array(dict_weeks_real[wk])
        if scaler:
            # retorna a escala compatível com os dados reais
            a = scaler.inverse_transform(a.reshape(-1,1)).reshape(a.shape)        
            real = scaler.inverse_transform(real.reshape(-1,1)).reshape(real.shape)
        media_somas = a.mean(axis=0)
        std = a.std(axis=0)        
        handle_r,  = axes[wk].plot(x, real, label='real', marker='o', ls='--')
        handles_s, = axes[wk].plot(x, media_somas, label='{}'.format(wks_name[wk]), color=colors[wk])    
        axes[wk].fill_between(x, media_somas-std, media_somas+std, alpha=0.3,color=colors[wk])
        axes[wk].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        if wk in lista_ylabel:
            axes[wk].set_ylabel('Bicicletas alugadas')
        if wk in lista_xlabel:
            axes[wk].set_xlabel('Horas')        
        lista_handles.append(handles_s)
    lista_handles = [handle_r] + lista_handles
    plt.legend(lista_handles, wks_name, bbox_to_anchor=bbox, loc='lower right')
    plt.suptitle("{}".format(figtitle))
    return fig, axes

###########
# Minicurso
###########

def plot_series(time=[], series=[], format="-", markers=[None,"o"], start=0, end=None, label=[], figsize=(10,5)):
    fig, ax = plt.subplots(figsize=figsize)
    for t, s, m, l in zip(time, series, markers, label):
        ax.plot(t[start:end], s[start:end], format, label=l, marker=m)
    ax.set_xlabel("Tempo")
    ax.set_ylabel("Valor")
    if len(label)>1:
        plt.legend(fontsize=14)

def trend(time, slope=0):
    return slope * time

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)