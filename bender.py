import pandas as pd
import datetime
import math
import warnings
warnings.filterwarnings("ignore")

""" Load Data """
ts = pd.read_csv('/Users/gabeheim/documents/repos/bender/source files/XRPBTC_15m.csv', index_col=False)#.set_index('Open time')
for time in ['Open time', 'Close time']:
    ts[time].divide(10000)
    ts[time] = pd.to_datetime(ts[time]/1000,unit='s')

data = ts
# data = data[7 * math.floor(len(ts) / 8):]


""" Label Data """
def labels(row, target_percent, stop_loss_percent):
    try:
        target_index = data.loc[row._name:][data.Close >= row.Close * (1 + target_percent)].iloc[0]._name
    except IndexError:
        target_index = data.index.max() + 5
    try:
        stop_loss_index = data.loc[row._name:][data.Close <= row.Close * (1 - target_percent)].iloc[0]._name
    except IndexError:
        stop_loss_index = data.index.max() + 5
    if target_index > stop_loss_index:
        return 0
    elif target_index < stop_loss_index:
        return 1
    return None

data['label'] = data.apply(lambda row: labels(row, .05, .01), axis=1)


""" Add Features to data dataframe """

import talib as ta

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
# ar, br, vr .. (26), bias 20


""" Min max scale, polarize, or take percentages of indicators as new features """

from sklearn.preprocessing import minmax_scale

indicators = ['sma_10','macd','macd_signal','macd_hist','cci_24','mom_10','roc_10','rsi_5','wnr_9','slowk','slowd','adosc']
min_mix_indicators = ['Volume', 'Number of trades','sma_10','rsi_5','wnr_9','slowk','slowd','adosc']
for indicator in min_mix_indicators:
    data[indicator + '_min_max'] = minmax_scale(data[indicator])

    
polarize_indicators = ['macd','macd_signal','macd_hist','cci_24','mom_10','roc_10', 'wnr_9','adosc']
for indicator in polarize_indicators:
    data[indicator + '_polarize'] = data[indicator] / abs(data[indicator])
    
percentage_indicators = ['sma_10','mom_10','roc_10','rsi_5','slowk','slowd']
for indicator in percentage_indicators:
    
    data[indicator + '_percentage'] = data[indicator].pct_change()  
    data[indicator + '_percentage'] = ( (data[indicator + '_percentage'] - data[indicator + '_percentage'].min()) / 
                                       (data[indicator + '_percentage'].max() - data[indicator + '_percentage'].min()) ) * (1 - -1) + -1 
    
    #minmax_scale(data[indicator].pct_change(), feature_range=(-1, 1))

    
""" Perform Recursive Feature Extension """

from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import numpy as np
#,'cci_24', 'rsi_5', 'wnr_9'
features = data.copy().drop(['Open','High','Low','Open time', 'Close time', 'Volume', 'Number of trades', 'label', 'mom_10_percentage', 
                             'roc_10_percentage'] + [indicator for indicator in list(indicators)], axis=1
                           ).replace([np.inf, -np.inf], np.nan).ffill().bfill()#,'slowd','adosc']
# .replace(np.nan, features.mean())[polarize_indicators].reset_index().drop(['index'], axis=1)#.to_numpy()#.reset_index()
# features.replace(np.nan, features.mean(), inplace=True)
# display(features.isnull().sum())
# for i, col in enumerate(features.columns):
#     print(features.columns[i:i+2])
X_train,X_test,y_train,y_test = train_test_split(features,data.label.fillna(0),test_size=0.3,random_state=100)

estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=10)#, step=1)
selector = selector.fit(X_train, y_train)
print(selector.ranking_)
X_train = X_train[[X_train.columns[i] for i, value in enumerate(selector.ranking_) if value == 1]]
X_test = X_test[[X_test.columns[i] for i, value in enumerate(selector.ranking_) if value == 1]]


for column in X_train.columns:
    X_train[column] = minmax_scale(X_train[column], feature_range=(-1, 1))
    X_test[column] = minmax_scale(X_test[column], feature_range=(-1, 1))

from sklearn.decomposition import PCA
pca = PCA(n_components=6)
x_train = pd.DataFrame(pca.fit_transform(X_train))
x_test = pd.DataFrame(pca.fit_transform(X_test))

def create_inout_sequences(input_data, input_labels, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = torch.FloatTensor(input_data[i:i+tw])
        train_label = torch.FloatTensor([input_labels[i+tw - 1]])
        inout_seq.append((train_seq ,train_label))
    return inout_seq

train_window = 12
train = create_inout_sequences(x_train.values[:-math.floor(len(x_train)/10)], list(y_train[:math.floor(-len(x_train)/10)]), train_window)
validate = create_inout_sequences(x_train.values[-math.floor(len(x_train)/10):], list(y_train[math.floor(-len(x_train)/10):]), train_window)
test = create_inout_sequences(x_test.values, list(y_test), train_window)


# import modules to build RunBuilder and RunManager helper classes
from collections import OrderedDict
from collections import namedtuple
from itertools import product
from IPython.display import clear_output
import sklearn
import itertools
import torch
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


import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

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
        self.data_sets = {
            'xrp_btc': {
                'train': train,
                'validate': validate,
                'test': test
            }
        }
        
        # tracking every epoch count, loss, accuracy, time
        self.epoch_count = 0
        self.epoch_loss = {'train': 0, 'validate': 0}
        self.epoch_num_correct = {'train': 0, 'validate': 0}
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

        # record model, loader and TensorBoard 
        self.network = None
        self.loaders = None
        self.tb = None        
        
    # record the count, hyper-param, model, loader of each run
    # record sample images and network graph to TensorBoard    
    def begin_run(self, run, network):

        self.run_start_time = time.time()

        self.run_params = run
        self.run_count += 1
#         self.run_plot_statistics[self.run_count] = {}

        self.network = network
        self.tb = SummaryWriter(comment=f'-hi')#{run}')

#         images, labels = next(iter(self.loaders['train']))
#         grid = torchvision.utils.make_grid(images)

#         self.tb.add_image('images', grid)
#         self.tb.add_graph(self.network, images.reshape(1, 784))

    # when run ends, close TensorBoard, zero epoch count
    def end_run(self):
        self.tb.close()
        self.epoch_count = 0

    # zero epoch count, loss, accuracy, 
    def begin_epoch(self, epoch_number):
        self.epoch_start_time = time.time()

        self.epoch_count = epoch_number
#         self.run_plot_statistics[self.run_count][self.epoch_count] = {
#             'loss': {phase: [] for phase in self.loaders.keys()},
#             'accuracy': {phase: [] for phase in self.loaders.keys()}
#         }
        self.epoch_loss = {'train': 0, 'validate': 0}
        self.epoch_num_correct = {'train': 0, 'validate': 0}

    def end_epoch(self):
        # calculate epoch duration and run duration(accumulate)
        epoch_duration = time.time() - self.epoch_start_time
        run_duration = time.time() - self.run_start_time

        # record epoch loss and accuracy
        loss = {phase: self.epoch_loss[phase] / (len(self.data_sets['xrp_btc'][phase])) for phase in ['train', 'validate']}
        accuracy = {phase: self.epoch_num_correct[phase] / (len(self.data_sets['xrp_btc'][phase])) for phase in ['train', 'validate']}

        # Record epoch loss and accuracy to TensorBoard 
        self.tb.add_scalar('Train Loss', loss['train'], self.epoch_count)
        self.tb.add_scalar('Validate Loss', loss['validate'], self.epoch_count)
        self.tb.add_scalar('Train Accuracy', accuracy['train'], self.epoch_count)
        self.tb.add_scalar('Validate Accuracy', accuracy['validate'], self.epoch_count)
        
        # Record params to TensorBoard
        for name, param in self.network.named_parameters():
            self.tb.add_histogram(name, param, self.epoch_count)
            self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch_count)
        
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
        self.epoch_num_correct[phase] += self._get_num_correct(outputs, labels)
#         try:
#             self.run_plot_statistics[self.run_count][self.epoch_count]['accuracy'][phase].append(self.epoch_num_correct[phase] / \
#                                                         len(self.run_plot_statistics[self.run_count][self.epoch_count]['accuracy']))
#         except: # if first image
#             self.run_plot_statistics[self.run_count][self.epoch_count]['accuracy'][phase].append(self.epoch_num_correct[phase])
        
    def track_test_predictions(self, prediction, label):
        self.test_predictions.append(prediction)
        self.test_labels.append(label)

    @torch.no_grad()
    def _get_num_correct(self, output, label):
        return 1 if int(torch.argmax(output)) == int(label.item()) else 0
    
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

        # print(cm)
        

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

        cnf_matrix = sklearn.metrics.confusion_matrix(self.test_labels, self.test_predictions)
        self.run_data[-1]['confusion_matrix'] = cnf_matrix

        # Plot normalized confusion matrix
        fig = plt.figure()
        fig.set_size_inches(7, 6, forward=True)
        #fig.align_labels()

        # fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
#         self.plot_confusion_matrix(cnf_matrix, classes=self.run_params.label_subset, normalize=True,
#                               title='Normalized confusion matrix')
        
        self.runs.append(self.run_data)
#         result_df = pd.DataFrame.from_dict(
#                 self.run_data[-1], 
#                 orient = 'columns',
#         )
#         display(result_df)
        

#         with open(f'results/{fileName}.json', 'w', encoding='utf-8') as f:
#             json.dump(self.run_data, f, ensure_ascii=False, indent=4)

class NeuralNet(nn.Module):
    def __init__(self, weight_init={'function':torch.nn.init.xavier_uniform}, hidden_neurons=100, output_neurons=2, 
                 hidden_activation=functional.relu, output_activation=torch.nn.Softmax(dim=2), input_size=6):
        super(NeuralNet, self).__init__()
        
        # hyper parameters
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        
        # layers
        self.hidden = nn.LSTM(input_size, hidden_neurons)  # input to hidden layer
#         if weight_init:
#             weight_init['function'](self.hidden.weight)
            
        self.out = nn.Linear(in_features=hidden_neurons, out_features=output_neurons) # hidden layer to output
        if weight_init:
            weight_init['function'](self.out.weight)
            
        self.hidden_cell = (torch.zeros(1,1,hidden_neurons),
                            torch.zeros(1,1,hidden_neurons))
        
    def forward(self, x):
        lstm_out, self.hidden_cell = self.hidden(x.view(len(x) ,1, -1), self.hidden_cell)
        predictions = self.out(lstm_out.view(len(x), -1))
        return predictions[-1]
    
#         h_pred = self.hidden_activation(self.hidden(x)) # h = dot(input,w1) 
#                                          #  and nonlinearity (relu)
            
#         return self.output_activation(self.out(h_pred.reshape(1, 1, self.hidden_neurons)))#torch.tensor([h_pred[0][0][0], h_pred[0][1][0]]))) #torch.from_numpy(result)


def sum_squared_error(out, label):
#     print('label, outputs: ', label, out)
#     result = (label - out) ** 2
#     print(result.sum())
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
    data_set = ['xrp_btc'],
    hidden_neurons = [100],#, 100, 5], #1
    
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
    validation_split = [0.1]
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


""" Train/Test """

import time, copy, json
from torch.utils.tensorboard import SummaryWriter
epochs = 10
# get all runs from params using RunBuilder class
# print(f"Runs: {RunBuilder.get_runs(params)}")
for run in RunBuilder.get_runs(params):

    # if params changes, following line of code should reflect the changes too
    net = NeuralNet(
        hidden_neurons=run.hidden_neurons,
        hidden_activation=run.hidden_activation, output_activation=run.loss_output['output_activation'])
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
            else:
                net.eval()   # Set model to evaluate mode
                
            # Iterate over data.
            for images, labels in m.data_sets['xrp_btc'][phase]:

                if count % 1000 == 0:
                    print(f'image #{count} {phase} {sum_loss / 1000} {m.epoch_num_correct[phase]}')
                    sum_loss = 0
                    
                net.hidden_cell = (torch.zeros(1, 1, net.hidden_neurons),
                    torch.zeros(1, 1, net.hidden_neurons))

                X = Variable(images)#.reshape(1, 784, 1).squeeze(0)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = net(X)         
                    
                    Y = np.zeros(2)
#                     Y[list(range(len(run.label_subset)))] = error_encoding_map[run.loss_output['output_activation']]['cold'](outputs)
                    Y[int(labels.item())] = 1 #error_encoding_map[run.loss_output['output_activation']]['hot'](outputs)
                    Y = Variable(torch.from_numpy(Y).long()).unsqueeze(0)
        
                    loss = criterion(outputs, Y)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                sum_loss += loss
                m.track_loss(phase, loss)
                m.track_num_correct(phase, outputs, labels)
                
#                 if count > 500:
#                     break
                count += 1
                    
        m.end_epoch()
    
    # Testing
    y_true = []
    y_predict = []
    phase = 'test'
    count = 0
    net.eval()
    for images, labels in m.data_sets['xrp_btc']['test']:

        X = Variable(images)#.unsqueeze(2)
        Y = Variable(labels)

        outputs = net(X)
        predicted_class = int(torch.argmax(outputs))
        
        m.track_test_predictions(predicted_class, labels.item())

        if count % 250 == 0:
            print(f'image #{count}: {outputs} {labels} {predicted_class}')
        count += 1

#         if count > 500:
#             break

    m.end_run()

    # when all runs are done, show results
    m.results('trial')
    
    
""" epoch vs accuracy/loss plot """

from pandas.plotting import table 
source_df = pd.DataFrame(m.run_data)
display(source_df)
record = {
    'xrp_btc': [],
}
data_set = 'xrp_btc'
    
df = source_df.loc[source_df.data_set == data_set]

for run_i in df['run'].unique():
    run_data = df.loc[df.run == run_i]
    epochs = run_data.epoch.values

    # Accuracy 1st y-axis
    fig, ax1 = plt.subplots()

    # Loss 2nd y-axis
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    colors = ['red', 'green', 'blue']

    for phase_i, phase in enumerate(['train', 'validate']):

        accuracy = run_data[f'{phase} accuracy'].values

        # record record
        if phase == 'validate':
            record[data_set].append({
                'max_accuracy': np.max(accuracy).round(3),
                'epoch': np.where(accuracy == np.max(accuracy))[0] + 1,

#                 'run': run_i
            })
#             for variable in variables:
#                 record[data_set][-1][variable] = run_data[variable].values[0]

        loss = run_data[f'{phase} loss'].values
        phase_accuracy, = ax1.plot(epochs, accuracy, 
             color=colors[phase_i],   
             linewidth=1.0
        )
        phase_accuracy.set_label(f"{phase.capitalize()} Accuracy")

        phase_loss, = ax2.plot(epochs, loss, 
             color=colors[phase_i],   
             linewidth=1.0,
             linestyle='--' 
        )
        phase_loss.set_label(f"{phase.capitalize()} Loss")

    ax1.legend(loc='lower left')
    ax2.legend(loc='lower right')

    x_label = "Epoch"
#     for variable in variables:
#         x_label += f"\n{variable} = {run_data[variable].values[0]}"
    ax1.set_xlabel(x_label)
    ax1.set_ylabel("Accuracy")
    ax2.set_ylabel("Loss")

    plt.title(f"{run_data['data_set'].values[0]}: Epoch vs. Accuracy and Loss")

    ax1.set_ybound(lower=0.1, upper=1)

    save_string = "sigma_relationship.png"
#     for variable in variables:
#         save_string = f"{data_set}_{variable}_{run_data[variable].values[0]}_" + save_string
#     plt.savefig(f"./{variables[0]}/" + save_string, bbox_inches='tight')
    plt.show()
#     break

