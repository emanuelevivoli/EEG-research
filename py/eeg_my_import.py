# Get Paths
from glob import glob

# EEG package
from mne import  pick_types
from mne.io import read_raw_edf

import os
import numpy as np


#%%
# Get file paths
PATH = '../data/' #'/Users/jimmy/data/PhysioNet/'#'/rigel/pimri/users/xh2170/data2/data/' #PATH = './data/'
SUBS = glob(PATH + 'S[0-9]*')
FNAMES = sorted([x[-4:] for x in SUBS])

REMOVE = ['S088', 'S089', 'S092', 'S100']

# Remove subject 'S089' with damaged data and 'S088', 'S092', 'S100' with 128Hz sampling rate (we want 160Hz)
FNAMES = [ x for x in FNAMES if x not in REMOVE] 

emb = {'T0': 1, 'T1': 2, 'T2': 3}


def my_get_data(data_type, subj_num=FNAMES, epoch_sec=0.0625):
    """ Import from edf files data and targets in the shape of 3D tensor
    
        Output shape: (Trial*Channel*TimeFrames)
        
        Some edf+ files recorded at low sampling rate, 128Hz, are excluded. 
        Majority was sampled at 160Hz.
        
        epoch_sec: time interval for one segment of mashes (0.0625 is 1/16 as a fraction)
    """
    
    # Event codes mean different actions for two groups of runs
    if data_type == 'Real':
        run_type_0 = '01'.split(',')
        run_type_1 = '03,07,11'.split(',')
        run_type_2 = '05,09,13'.split(',')
    else:
        run_type_0 = '02'.split(',')
        run_type_1 = '04,08,12'.split(',')
        run_type_2 = '06,10,14'.split(',')
    
    # Initiate X, y
    X = []
    y = []
    p = []
    dim = dict()
    
    # To compute the completion rate
    count = len(subj_num)
    
    # fixed numbers
    nChan = 64 
    sfreq = 160
    sliding = epoch_sec/2 
    timeFromQue = 0.5
    timeExercise = 4.1 #secomds
    magic_number = 51
    
    run_0_segments = int(magic_number * (magic_number*timeFromQue))
    run_segments = magic_number


    # Sub-function to assign X and X, y
    def append_X(n_segments, data, event=[]):
        # Data should be changed
        '''This function generate a tensor for X and append it to the existing X'''

        def window(n):
            # (80) + (160 * 1/16 * n) 
            windowStart = int(timeFromQue*sfreq) + int(sfreq*sliding*n) 
            # (80) + (160 * 1/16 * (n+2))
            windowEnd = int(timeFromQue*sfreq) + int(sfreq*sliding*(n+2)) 
            
            while (windowEnd - windowStart) != sfreq*epoch_sec:
                windowEnd += int(sfreq*epoch_sec) - (windowEnd - windowStart)
                
            return [windowStart, windowEnd]
        
        new_x = []
        for n in range(n_segments):
            # print('data[:, ',window(n)[0],':',window(n)[1],'].shape = ', data[:, window(n)[0]:window(n)[1]].shape, '(',nChan,',',int(sfreq*epoch_sec),')')
            
            if data[:, window(n)[0]:window(n)[1]].shape==(nChan, int(sfreq*epoch_sec)):
                new_x.append(data[:, window(n)[0]: window(n)[1]])
                    
        return new_x
    
    def append_X_Y(p, run_type, event, old_x, old_y, old_p, data):
        '''This function seperate the type of events 
        (refer to the data descriptitons for the list of the types)
        Then assign X and Y according to the event types'''
        # Number of sliding windows

        # print('data', data.shape[1])
        n_segments = run_segments
        #n_segments = floor(data.shape[1]/(epoch_sec*sfreq*timeFromQue) - 1/epoch_sec - 1)
        # print('run_'+str(run_type),' n_segments', n_segments, 'data', data.shape)
        
        # Rest excluded
        if event[2] == emb['T0']:
            return old_x, old_y, old_p
        
        # y assignment
        if run_type == 1:
            temp_y = [1] if event[2] == emb['T1'] else [2]
        
        elif run_type == 2:
            temp_y = [3] if event[2] == emb['T1'] else [4]
            
        # print('event[2]', event[2], 'run_type', run_type, 'temp_y', temp_y)            
        
        # print('timeExercise * sfreq', timeExercise*sfreq, ' ?= 656')
        new_x = append_X(n_segments, data, event)
        new_y = old_y + temp_y*len(new_x)
        new_p = old_p + p*len(new_x)
        
        return old_x + new_x, new_y, new_p
    
    # Iterate over subj_num: S001, S002, S003, ...
    for i, subj in enumerate(subj_num):
        # print('subj', subj)

        
        # Return completion rate
        if i%((len(subj_num)//10)+1) == 0:
            print('\n')
            print('working on {}, {:.0%} completed'.format(subj, i/count))
            print('\n')
        
        old_size = np.array(y).shape[0]
        # print('subj:', subj, '| y.shape', np.array(y).shape ,'| X.shape', np.array(X).shape)

        # Get file names
        fnames = glob(os.path.join(PATH, subj, subj+'R*.edf'))
        # Hold only the files that have an even number
        fnames = sorted([name for name in fnames if name[-6:-4] in run_type_0+run_type_1+run_type_2])

        # for each of ['02', '04', '06', '08', '12', '14']
        for i, fname in enumerate(fnames):
            # print('fname', fname)
            
            # Import data into MNE raw object
            raw = read_raw_edf(fname, preload=True, verbose=False)
            
            picks = pick_types(raw.info, eeg=True)
            # print('n_times', raw.n_times)
            
            if raw.info['sfreq'] != 160:
                print('{} is sampled at 128Hz so will be excluded.'.format(subj))
                break
            
            
            # High-pass filtering
            raw.filter(l_freq=1, h_freq=None, picks=picks)

            # Get annotation
            try:
                events = events_from_annotations(raw, verbose=False)
            except:
                continue

            # Get data
            data = raw.get_data(picks=picks)

            # print('event.shape', np.array(events[0]).shape, '| data.shape', data.shape)

            # Number of this run
            which_run = fname[-6:-4]

            """ Assignment Starts """ 
            # run 1 - baseline (eye closed)
            if which_run in run_type_0:

                # Number of sliding windows
                n_segments = run_0_segments
                # n_segments = floor(data.shape[1]/(epoch_sec*sfreq*timeFromQue) - 1/epoch_sec - 1)
                # print('run_0 n_segments', n_segments, 'data', data.shape)

                # Append 0`s based on number of windows
                new_X = append_X(n_segments, data)
                X += new_X
                y.extend([0] * len(new_X))
                p.extend([subj]* len(new_X))
                # print(events[0])   

            # run 4,8,12 - imagine opening and closing left or right fist    
            elif which_run in run_type_1:

                for i, event in enumerate(events[0]):

                    X, y, p = append_X_Y([subj], run_type=1, event=event, old_x=X, old_y=y, old_p=p, data=data[:, int(event[0]) : int(event[0] + timeExercise*sfreq)])
                    # print(event)   

            # run 6,10,14 - imagine opening and closing both fists or both feet
            elif which_run in run_type_2:

                for i, event in enumerate(events[0]):      

                    X, y, p = append_X_Y([subj], run_type=2, event=event, old_x=X, old_y=y, old_p=p, data=data[:, int(event[0]) : int(event[0] + timeExercise*sfreq)])
                    # print(event)    

        print('subj:', subj, '|', np.array(y).shape[0] - old_size, '| y.shape', np.array(y).shape ,'| X.shape', np.array(X).shape, '| p.shape', np.array(p).shape)
        dim[subj] =  np.array(y).shape[0] - old_size
    print(np.array(X).shape)

    X = np.stack(X)
    y = np.array(y).reshape((-1,1))
    p = np.array(p).reshape((-1,1))
    return X, y, p, dim