# import modules to build RunBuilder and RunManager helper classes
import pandas as pd
import datetime
import math
import warnings
warnings.filterwarnings("ignore")
from functools import partial
import multiprocessing
import time
import talib as ta
from sklearn.preprocessing import minmax_scale
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import random
import torch
import time, copy, json

from collections import OrderedDict
from collections import namedtuple
from itertools import product
from IPython.display import clear_output
import sklearn
import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as functional
from torch.autograd import Variable
from torch.autograd import Function
import torch.optim as optim
import numpy as np
import matplotlib
import pandas as pd
from sklearn import metrics
from functools import partial


import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

INVALID_LABEL = 99

def get_labels(row, data=None, train_window=None, target_percent=.01, stop_loss_percent=.005, mode='since3'):
    if mode == 'since':
        try:
            target_index = data.loc[row[6]:][data.Close >= row[4] * (1 + target_percent)].iloc[0]._name
        except IndexError:
            target_index = data.index.max() + 5
        try:
            stop_loss_index = data.loc[row[6]:][data.Close <= row[4] * (1 - stop_loss_percent)].iloc[0]._name
        except IndexError:
            stop_loss_index = data.index.max() + 5
        if target_index > stop_loss_index:
            return 0
        elif target_index < stop_loss_index:
            return 1
        return None
    if mode == 'average':
        mean = data.Close[row[6]:row[6]+10].mean()
#         print(mean)
        if mean - row[4] > 0: #row[4] * (1 + target_percent):
            return 1
#         elif mean <= row[4] * (1 - stop_loss_percent):
#             return -1
        else:
            return 0
        return None
    if mode == 'next':
        try:
            next = data.Close.values[row[6]+1]
        except:
            next = 0
#         print(next)
        if next > row[4]: #row[4] * (1 + target_percent):
            return 1
#         elif mean <= row[4] * (1 - stop_loss_percent):
#             return -1
        else:
            return 0
        return None
    if mode == 'since3':
#         max_index = data.High[row[6]:row[6]+21].idxmax()
#         min_index = data.Low[row[6]:row[6]+21].idxmin()
        try:
#             target_index = data.High[row[6]:row[6]+train_window+1].loc[data.High >= row[4] * (1 + target_percent)].index[0]
            target_index = data.Close[row[6]:row[6]+train_window+1].loc[data.Close >= row[4] * (1 + target_percent)].index[0]
        except IndexError:
            target_index = data.index.max() + 5
        try:
#             stop_loss_index = data.Low[row[6]:row[6]+train_window+1].loc[data.Low <= row[4] * (1 - stop_loss_percent)].index[0]
            stop_loss_index = data.Close[row[6]:row[6]+train_window+1].loc[data.Close <= row[4] * (1 - stop_loss_percent)].index[0]
        except IndexError:
            stop_loss_index = data.index.max() + 5
        if target_index > stop_loss_index:
            return 0
        elif target_index < stop_loss_index:
            return 2
        return 1
    
def normalize_sequence_columns(sequence_label, range=(-1, 1), indices=None):
    sequence, label = sequence_label

    input_df = pd.DataFrame(sequence, dtype=np.float32)
    for index in indices:
        input_df[index] = minmax_scale(input_df[index], feature_range=range)
    seq = torch.FloatTensor(input_df.values)
    label = torch.FloatTensor([label])
    return (seq, label)

# Read in the hyper-parameters and return a Run namedtuple containing all the 
# combinations of hyper-parameters
class RunBuilder():
    @staticmethod
    def get_runs(params):

        Run = namedtuple('Run', params.keys())

        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))

        return runs

class RunManager():
    def __init__(self):
        
        def view(image):
            return image.view(28*28)

        compose_transforms = [
            transforms.ToTensor(),
            view
        ]
        
        # Data sets to choose from
#         self.data_sets = {
# #             'sp500': {
# #                 'train': data_sets['sp500']['train'],
# #                 'validate': data_sets['sp500']['validate'],
# #                 'test': data_sets['sp500']['test']
# #             },
#             'BTCUSDT': {
#                 'train': data_sets['BTCUSDT']['train'],
#                 'validate': data_sets['BTCUSDT']['validate'],
#                 'test': data_sets['BTCUSDT']['test']
#             }
#         }
        self.data_sets = {
            'sp500': {
                'data': pd.read_csv(f'/Users/gabeheim/documents/concatenated_price_data/sp500.csv', index_col=False).drop(['Adj Close'], axis=1)
            },
            'BTCUSDT': {
                'data': pd.read_csv(f'/Users/gabeheim/documents/concatenated_price_data/BTCUSDT.csv', index_col=False)
            }
        }

        start_r = 180000

        self.data_sets['BTCUSDT']['data'] = self.data_sets['BTCUSDT']['data'][start_r:start_r+20000].reset_index().drop(['index'], axis=1)
        
        self.global_labels = None
        
        # tracking every epoch count, loss, accuracy, time
        self.epoch_count = 0
        self.epoch_loss = {'train': 0, 'validate': 0}
        self.epoch_num_correct = None
        self.epoch_start_time = None

        # tracking every run count, run data, hyper-params used, time
        self.run_params = None
        self.run_count = 0
        self.run_data = []
        self.run_start_time = None
        self.runs = pd.DataFrame()
#         self.run_plot_statistics = {}
        
        # testing data
        self.test_predictions = []
        self.test_labels = []
        self.test_correct_count = None

        # record model, loader and TensorBoard 
        self.network = None
        print("Run manager initialized")
        
    def create_inout_sequences(self, input_data, input_labels, tw):
        inout_seq = []
        L = len(input_data)
        for i in range(L-tw):
            train_seq = torch.FloatTensor(input_data[i:i+tw])
            train_label = torch.FloatTensor([input_labels[i+tw - 1]])
            if train_label == INVALID_LABEL or torch.isnan(train_seq).any():
                continue
            inout_seq.append((train_seq ,train_label))
        return inout_seq

    def get_sequenced_train_val_test(self, features, datad, train_window=20):
        data = datad['data']
        labels = datad['labels']
        all = self.create_inout_sequences(features.values, list(labels), train_window)
        random.shuffle(all)
    #     print(all)
        train = all[:-math.floor(len(features)/2.5)]
        validate = all[-math.floor(len(features)/2.5):-math.floor(len(features)/4)]
        test = all[-math.floor(len(features)/4):]

        tdf = pd.DataFrame(train)
        print(tdf[0:1])
        print(tdf[1].values[0])

        all_labels = [x for x in data.label.unique() if str(x) != 'nan']

        label_rows = {}
        for label in all_labels:
            print(label, len(tdf.loc[tdf[1] == torch.tensor([label])]))
            label_rows[label] = tdf.loc[tdf[1] == torch.tensor([label])]


        min_len = min(len(v) for k, v in label_rows.items())
        min_key = [label for label in all_labels if len(label_rows[label]) == min_len]
        print()
        for label in all_labels:
            to_remove = np.random.choice(label_rows[label].index,size=len(label_rows[label]) - min_len,replace=False)
            tdf = tdf.drop(to_remove)
            print(label, len(tdf.loc[tdf[1] == torch.tensor([label])]))
        print()

        train = [tuple(r) for r in tdf.to_numpy()]
        print(len(train), len(validate), len(test))
        return train, validate, test
    
    
    def get_pca_features(self, features, pca_components=10):

        pca = PCA(n_components=pca_components)
        features_pca = pd.DataFrame(pca.fit_transform(features))
        return features_pca

    def rfe(self, features, labels, sample_size=2000, trials=4, select=10):
        estimator = SVR(kernel="linear")
        selector = RFE(estimator, n_features_to_select=1)#, step=1)
        sums = [0] * len(features.columns)
        for trial in range(trials):
            samples = features.sample(n=sample_size)
            selector = selector.fit(samples, labels.loc[samples.index])
            print(selector.ranking_)
            sums = [sum(i) for i in zip(sums, selector.ranking_)]
            print(sums)
            print()
        return features[[features.columns[i] for i in np.argsort(sums)[-select:] ]]
    
    def prepare_data(self, run):
        
        name = run.data_set
        datad = self.data_sets[name]
        data = datad['data']
        train_window = run.train_window
        n = 2
        pool = multiprocessing.Pool(math.floor(multiprocessing.cpu_count() / 3))
        print("using ", math.floor(multiprocessing.cpu_count() / 3), "cpus")
        pca_components = run.pca_components
        label_mode = run.label_mode['mode']
        target_percent = run.label_mode['target_percent']
        stop_loss_percent = run.label_mode['stop_loss_percent']
        rfe_select = run.rfe_select
        chosen_dependent = run.chosen_dependent
        
        # labels
        print('\n Getting labels \n')
        
        data['index'] = data.index
        # too volatile class?
        n = 0

        start = time.time()
        data['label'] = pool.map(partial(get_labels, data=data, train_window=train_window, mode=label_mode,
                                       target_percent=target_percent, stop_loss_percent=stop_loss_percent), 
                                 [tuple(r) for r in data.to_numpy()] )  # process data_inputs iterable with pool
        print(name, 'pool label took: ', time.time() - start)
        self.global_labels = [x for x in sorted(self.data_sets[run.data_set]['data'].label.unique()) if str(x) != 'nan']
        self.epoch_num_correct = {'train': {k: 0 for k in self.global_labels},
                                  'validate': {k: 0 for k in self.global_labels}}
        self.test_correct_count = {k: 0 for k in self.global_labels}
            
        # log class distribution
        print('\n Class distribution: \n')
        
        for label in sorted(data.label.unique()):
            print(name, label, len(data.loc[data.label == label]))
                
        # get indicators
        print('\n Getting indicators \n')
        
        data['sma_10'] = ta.SMA(data.Close, timeperiod=10)
        macd, macd_signal, macd_hist = ta.MACDFIX(data.Close, signalperiod=9)
        data['macd'] = macd
        data['macd_signal'] = macd_signal
        data['macd_hist'] = macd_hist
        data['cci_24'] = ta.CCI(data.High, data.Low, data.Close, timeperiod=24)
        data['mom_10'] = ta.MOM(data.Close, timeperiod=10)
        data['roc_10'] = ta.ROC(data.Close, timeperiod=10)
        data['rsi_5'] = ta.RSI(data.Close, timeperiod=5)
        data['wnr_9'] = ta.WILLR(data.High, data.Low, data.Close, timeperiod=9)
        slowk, slowd = ta.STOCH(data.High, data.Low, data.Close, fastk_period=5, 
                                slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
        data['slowk'] = slowk
        data['slowd'] = slowd
        data['adosc'] = ta.ADOSC(data.High, data.Low, data.Close, data.Volume, fastperiod=3, slowperiod=10)
        data = data[30:].reset_index()
        data = data.drop(['level_0'], axis=1)
        data['index'] = data.index
            
        # min max them
        print('\n Min-max scaling indicators \n')
        independent_indicators = ['macd', 'macd_signal', 'macd_hist', 'cci_24', 'mom_10', 'roc_10','rsi_5','wnr_9','slowk','slowd','adosc']
        for indicator in independent_indicators:
            name = indicator + '_min_max'
            mean = data[indicator].mean()
            std = data[indicator].std()
            data[indicator].loc[data[indicator] > mean + 3 * std] = mean + 3 * std
            data[indicator].loc[data[indicator] < mean - 3 * std] = mean - 3 * std
            data[name] = (data[indicator] - mean) / std 
            data[name] = minmax_scale(data[indicator], feature_range=(-1,1))
                
        # percentage them 
        print('\n Getting percentage fluctuation of indicators \n')
        percentage_indicators = ['Close', 'Volume', 'sma_10'] + independent_indicators
        for indicator in percentage_indicators:
            name = indicator + '_percentage'
            data[name] = data[indicator].pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
            mean = data[name].mean()
            std = data[name].std()
            data[name].loc[data[name] > mean + 3 * std] = mean + 3 * std
            data[name].loc[data[name] < mean - 3 * std] = mean - 3 * std
            data[name] = (data[name] - mean) / std 
            data[name] = minmax_scale(data[name], feature_range=(-1, 1))
                
        # isolate features / labels, fillnas
        print('\n Isolating features and labels, filling their nas \n')
        dependent_indicators = ['Open','High','Low','Close','Volume', 'sma_10']

        datad['features'] = data.copy().drop(['index', 'Date', 'label'] + dependent_indicators + independent_indicators, axis=1
                                   ).fillna(0).replace([np.inf, -np.inf], np.nan).ffill()

        print(datad['features'][0:2])

        datad['labels'] = data['label'].copy().replace([np.inf, -np.inf], np.nan).fillna(INVALID_LABEL)
        features = datad['features']
        labels = datad['labels']
        print(datad['labels'][0:2])
            
        # rfe
        if rfe_select > 0:
            print('\n Performing RFE \n')
            datad['rfe_features'] = self.rfe(features, labels, sample_size=2000, trials=4, select=rfe_select)
            print(datad['rfe_features'][0:2])
            rfe_features = datad['rfe_features']
            
        # pca
        if pca_components:
            print('\n Performing PCA \n')
            datad['rfe_pca_features'] = get_pca_features(rfe_features, pca_components=7)
            print(datad['rfe_pca_features'][0:2])
            
        # sequence and split to train val test
        print('\n Building sequences and splitting to train, val, and test sets \n')
        if rfe_select > 0:
            chosen_indicators = rfe_features #features_pca
            combined = pd.concat([data[chosen_dependent], chosen_indicators], axis=1)
        else:
            print("WORKS")
            print(data[chosen_dependent][0:2])
            combined = data[chosen_dependent]
        print(combined[0:2])
        print(len(labels))
        train, validate, test = self.get_sequenced_train_val_test(combined, datad, train_window=train_window)
        datad['train'] = train
        datad['validate'] = validate
        datad['test'] = test
        print(train[0])
        
        # normalize certain features within sequence
        print('\n Normalizing certain features within sequences \n')
        attempts = 3
        train = datad['train']
        validate = datad['validate']
        test = datad['test']

        for a in range(attempts):
            try:
                start = time.time()
                train = pool.map(partial(normalize_sequence_columns, indices=list(range(len(chosen_dependent)))), train.copy())
                print(len(train), 'train took:', time.time() - start)
                break
            except RuntimeError:
                print('train err ', a)
                pass

        for a in range(attempts):
            try:
                start = time.time()
                validate = pool.map(partial(normalize_sequence_columns, indices=list(range(len(chosen_dependent)))), validate.copy())
                print(len(validate), 'val took:', time.time() - start)
                break
            except RuntimeError:
                print('val err ', a)
                pass

        for a in range(attempts):
            try:
                start = time.time()
                test = pool.map(partial(normalize_sequence_columns, indices=list(range(len(chosen_dependent)))), test.copy())
                print(len(test), 'test took:', time.time() - start)
                break
            except RuntimeError:
                print('test err ', a)
                pass
        print(train[0])
            
        pool.close()
        
    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard    
    def begin_run(self, run, network):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1
#         self.run_plot_statistics[self.run_count] = {}
    
        self.network = network
        
        self.prepare_data(run)

    # when run ends, close TensorBoard, zero epoch count
    def end_run(self, net):
        self.epoch_count = 0
        
        self.run_data[-1]['net'] = net
        
        test_accuracy = sum([v for k, v in self.test_correct_count.items()]) / (len(self.data_sets[self.run_params.data_set][phase]))
        self.run_data[-1]['test_accuracy'] = test_accuracy
        
        cnf_matrix = sklearn.metrics.confusion_matrix(self.test_labels, self.test_predictions)
        self.run_data[-1]['confusion_matrix'] = cnf_matrix
        
        self.runs.append(self.run_data)
        print("RUN RESULTS:")
        print(self.run_data)
        
        

        # Plot normalized confusion matrix
#         fig = plt.figure()
#         fig.set_size_inches(7, 6, forward=True)
        #fig.align_labels()

        # fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
#         self.plot_confusion_matrix(cnf_matrix, classes=self.run_params.label_subset, normalize=True,
#                               title='Normalized confusion matrix')
        

    # zero epoch count, loss, accuracy, 
    def begin_epoch(self, epoch_number):
        self.epoch_start_time = time.time()

        self.epoch_count = epoch_number
#         self.run_plot_statistics[self.run_count][self.epoch_count] = {
#             'loss': {phase: [] for phase in self.loaders.keys()},
#             'accuracy': {phase: [] for phase in self.loaders.keys()}
#         }
        self.epoch_loss = {'train': 0, 'validate': 0}
        self.epoch_num_correct = {'train': {k: 0 for k in self.global_labels},
                                  'validate': {k: 0 for k in self.global_labels}}

    def end_epoch(self):
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # record epoch loss and accuracy
        def get_all_correct(dict):
            return sum([v for k, v in dict.items()])
        loss = {phase: self.epoch_loss[phase] / (len(self.data_sets[self.run_params.data_set][phase])) for phase in ['train', 'validate']}
        accuracy = {phase: get_all_correct(self.epoch_num_correct[phase]) / (len(self.data_sets[self.run_params.data_set][phase])) for phase in ['train', 'validate']}
        
        # Write into 'results' (OrderedDict) for all run related data
        results = OrderedDict()
        results['run'] = self.run_count
        results['epoch'] = self.epoch_count
        results['train loss'] = loss['train']
        results['validate loss'] = loss['validate']
        results['train accuracy'] = accuracy['train']
        results['validate accuracy'] = accuracy['validate']
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration

        # Record hyper-params into 'results'
        for parameter, value in self.run_params._asdict().items(): 
            if type(value) == dict:
                for true_parameter, true_value in value.items():
                    results[true_parameter] = true_value
                continue
                
            results[parameter] = value
            
        self.run_data.append(results)

#         print(results)

    # accumulate loss of batch into entire epoch loss
    def track_loss(self, phase, raw_loss):
        loss = raw_loss.item() 
        self.epoch_loss[phase] += loss
        
#         self.run_plot_statistics[self.run_count][self.epoch_count]['loss'][phase].append(loss)

    # accumulate number of corrects of batch into entire epoch num_correct
    def track_num_correct(self, phase, outputs, labels):
        try:
            l_i = int(labels.item())
        except:
            l_i = 0
        self.epoch_num_correct[phase][l_i] += self._get_num_correct(outputs, labels)
#         try:
#             self.run_plot_statistics[self.run_count][self.epoch_count]['accuracy'][phase].append(self.epoch_num_correct[phase] / \
#                                                         len(self.run_plot_statistics[self.run_count][self.epoch_count]['accuracy']))
#         except: # if first image
#             self.run_plot_statistics[self.run_count][self.epoch_count]['accuracy'][phase].append(self.epoch_num_correct[phase])
        
    def track_test_predictions(self, prediction, label):
        self.test_predictions.append(prediction)
        self.test_labels.append(label)
        try:
            l_i = int(labels.item())
        except:
            l_i = 0
        self.test_correct_count[l_i] += 1 if prediction == label else 0

    @torch.no_grad()
    def _get_num_correct(self, output, label):
        try:
            l_i = int(labels.item())
        except:
            l_i = 0
        return 1 if int(torch.argmax(output)) == l_i else 0
    
    def plot_confusion_matrix(self, cm, classes, variables, normalize=False, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
                
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print(f"Normalized Confusion Matrix (Run #{len(self.runs) - 1})")
        else:
            print('Confusion matrix, without normalization')        

        ax = plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(f'{df_row.data_set}: Confusion matrix')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=90)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        
        x_label = "Predicted label"
        for variable in variables:
            x_label += f"\n{variable} = {run_data[variable].values[0]}"
        ax.set_xlabel(x_label)
        plt.show()
    
    # save end results of all runs into json for further analysis
    def results(self, fileName):
        return
        
#         result_df = pd.DataFrame.from_dict(
#                 self.run_data, 
#                 orient = 'columns',
#         )
#         print(result_df)
        

#         with open(f'results/{fileName}.json', 'w', encoding='utf-8') as f:
#             json.dump(self.run_data, f, ensure_ascii=False, indent=4)

class NeuralNet(nn.Module):
    def __init__(self, weight_init={'function':torch.nn.init.xavier_uniform}, hidden_neurons=128, output_neurons=None, 
                 hidden_activation=functional.relu, output_activation=torch.nn.Softmax(dim=2), input_size=None,
                dropout_p=None):
        super(NeuralNet, self).__init__()
        
        # hyper parameters
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.input_size = input_size
        
        # layers
        self.lstm = nn.LSTM(input_size, hidden_neurons)
        self.dropout = nn.Dropout(p=dropout_p)
        
        self.hidden_cell = (torch.zeros(1,1,hidden_neurons),
                            torch.zeros(1,1,hidden_neurons))
        
#         self.cnn = [#nn.Sequential(
#             nn.Conv1d(input_size, 128, 1),
#             nn.Conv1d(in_channels=128, out_channels=256, kernel_size=8, stride=1),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Conv1d(256, 128, kernel_size=5, stride=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU(),
#             nn.Conv1d(128, 128, kernel_size=3, stride=1),
#             nn.BatchNorm1d(128),
#             nn.ReLU()#,
# #             nn.AvgPool1d(kernel_size=7, stride=1,padding=0) #(Lin+2*p-k)/s+1
#         ]#)
        
        self.out = nn.Linear(in_features=hidden_neurons, out_features=output_neurons) # hidden layer to output
        if weight_init:
            weight_init['function'](self.out.weight)
            
        
        
    def forward(self, x, count):
#         print('raw input size', x.size())
#         print('lstm altered size', x.view(len(x) ,1, self.input_size).size())
        lstm_input = x.view(len(x) ,1, self.input_size)
        lstm_out, self.hidden_cell = self.lstm(lstm_input, self.hidden_cell)
        lstm_out = self.dropout(lstm_out)
#         if count % 50 == 0:
#             print('lstm out:', lstm_out.size())
#             print('lstm out out:', self.out(lstm_out).size())#, self.out(lstm_out))
#             print('lstm activation:', self.output_activation(self.out(lstm_out)).size(), self.output_activation(self.out(lstm_out))[-1])
#         predictions = self.output_activation(self.out(lstm_out))
#         return predictions[-1]

# #         print('cnn altered size', x.reshape(1, self.input_size, len(x)).size())
#         cnn_out = x.reshape(1, self.input_size, len(x))
#         for i, layer in enumerate(self.cnn):
# #             print(i, '\nx: ', cnn_out.size())
#             cnn_out = layer(cnn_out)
# #             print('result: ', cnn_out.size(), '\n')
#         cnn_out = cnn_out.reshape(-1, 1, 128)
    
    
#         print(lstm_out.size(), cnn_out.size())
        features = lstm_out#torch.cat((lstm_out, cnn_out))
#         print(features.size())
        result = self.out(features)
#         print('res', result.size())
        result = self.output_activation(result)
#         print('res act', result.size())
#         print(result[-1])
        return result[-1]
        

def sum_squared_error(out, label):
#     print(label, out)
#     print(label - out)
#     print()
    return ((label - out) ** 2).sum()

def mean_squared_error(outputs, labels):
    return sum_squared_error(outputs, labels) / len(outputs)

def cross_entropy(outputs, labels):
    return -1 * (torch.log(outputs) * labels + (torch.log(1 - outputs)) * (1 - labels)).sum()

def dummy_activation(x):
    return x

# put all hyper params into a OrderedDict, easily expandable
params = OrderedDict(
    data_set = ['sp500'],#'BTCUSDT'], #'sp500', 
    hidden_neurons = [50],#, 100, 5], #1
    
    batch_size = [1],
    
    weight_init = [{
        'function': torch.nn.init.xavier_uniform,
        'name': "Xavier Uniform"
    }],
    
    hidden_activation = [torch.relu],#, torch.tanh, torch.relu],
    loss_output = [
        {
        'criterion': sum_squared_error,
        'output_activation': torch.nn.Softmax(dim=2)
    },  

    ],
    
    learning_rate = [0.01],#, .001],
    momentum = [0.1],#, 0],
    optimizer = [optim.SGD],#, optim.Adam], #optim.Adam(network.parameters(), lr=run.lr)
    validation_split = [0.1],
    
    
    train_window = [10],
    pca_components = [None],#10
    label_mode = [
#         {
#         'mode': 'since3',
#         'label_count': 3,
#         'target_percent': .03, 
#         'stop_loss_percent': .03
#     },
#         {
#         'mode': 'average',
#         'label_count': 2,
#         'target_percent': None, 
#         'stop_loss_percent': None
#     },
        {
        'mode': 'next',
        'label_count': 2,
        'target_percent': None, 
        'stop_loss_percent': None
    }
    ],
    rfe_select = [3],
    chosen_dependent = [ ['Close', 'Volume', 'sma_10'] ],
    dropout_p = [.4]
)


def negative_one(x):
    return -1

def zero(x):
    return 0

def one(x):
    return 1

def argmax(x):
    return x[0][0][torch.argmax(x)].item()

error_encoding_map = {
    torch.sigmoid: {
        'cold': zero,
        'hot': one
    },
    torch.relu: {
        'cold': zero,
        'hot': one
    },
    torch.tanh: {
        'cold': negative_one,
        'hot': one
    },
    torch.nn.Softmax(dim=2): {
        'cold': zero,
        'hot': one
    },
    dummy_activation: {
        'cold': zero,
        'hot': argmax
    }
}

m = RunManager()



epochs = 3
# get all runs from params using RunBuilder class
# print(f"Runs: {RunBuilder.get_runs(params)}")
for run in RunBuilder.get_runs(params):
    print(run)
    # if params changes, following line of code should reflect the changes too
#     len(m.data_sets[run.data_set]['train'][0][0][0])

    use_last_data = False
    
    if not use_last_data:
        input_size = len(run.chosen_dependent) + run.rfe_select

        net = NeuralNet(input_size=input_size, output_neurons=run.label_mode['label_count'],
                       dropout_p=run.dropout_p)
    #         hidden_neurons=run.hidden_neurons,
    #         hidden_activation=run.hidden_activation, output_activation=run.loss_output['output_activation'])
        optimizer = run.optimizer(net.parameters(), lr=run.learning_rate, momentum=run.momentum)#copy.deepcopy(run.optimizer)

        sum_loss = 0
        criterion = run.loss_output['criterion']

    
        m.begin_run(run, net)
    
    # Training
    for epoch in range(epochs):
    
        m.begin_epoch(epoch + 1)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'validate']:
            count = 0
            if phase == 'train':
                net.train()  # Set model to training mode
#             else:
#                 net.eval()   # Set model to evaluate mode
                
            # Iterate over data.
            for images, labels in m.data_sets[m.run_params.data_set][phase]:

                if count % 1000 == 0:
                    print(f'sample #{count} {phase} {sum_loss / 1000} {m.epoch_num_correct[phase]}')
                    sum_loss = 0
                    
                net.hidden_cell = (torch.zeros(1, 1, net.hidden_neurons),
                    torch.zeros(1, 1, net.hidden_neurons))

                X = Variable(images)#.reshape(1, 784, 1).squeeze(0)

                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(X, count)         
                    Y = np.zeros(len(m.global_labels))
                    try:
                        l_i = int(labels.item())
                    except:
                        l_i = 0
                    try:
                        Y[l_i] = 1 
                    except:
                        print('bad label')
                        Y[0] = 1
                    Y = Variable(torch.from_numpy(Y).long()).unsqueeze(0)
                    loss = criterion(outputs, Y)
#                     if math.isnan(loss):
#                         print(outputs, Y)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                sum_loss += loss
                m.track_loss(phase, loss)
                m.track_num_correct(phase, outputs, labels)
                
#                 print('\n')
                count += 1
#                 if count > 500:
#                     break
                
                    
        m.end_epoch()
    
    # Testing
    y_true = []
    y_predict = []
    phase = 'test'
    count = 0
#     net.eval()
    for images, labels in m.data_sets[m.run_params.data_set]['test']:
#         print('sample')
        with torch.set_grad_enabled(False):
            X = Variable(images)#.unsqueeze(2)
            Y = Variable(labels)
            outputs = net(X, count)
            predicted_class = int(torch.argmax(outputs))

            m.track_test_predictions(predicted_class, labels.item())

        if count % 250 == 0:
            print(f'sample #{count}: {outputs} {labels} {predicted_class}')
        count += 1

#         if count > 500:
#             break
#         print('\n\n')

    if not use_last_data:
        m.end_run(net)
    else:
        break
# when all runs are done, show results
m.results('trial')