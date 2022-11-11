#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:46:57 2020

@author: Johannes Gäding
"""

import numpy
import matplotlib.pyplot as plt
import pandas
import pickle
import scipy.ndimage
import bz2
import multiprocessing as mp
from scipy.ndimage.interpolation import shift

#---------------------------------------------------
# Class for trajectories in .xyz-fileformat.
# Supports variable number of atoms per frame.
#
# Class is iterable and subscriptable by the n-frames
#
# functions:
#
# <xyz_trajectory>.positions()   	-- xyz-atom coords of selected frame - numpy.array(3,N)
# <xyz_trajectory>.atom_types()	-- atom-types of slected frame - numpy.array(1,N) - floats
#
# porperties:
#
# <xyz_trajectory>.frame		-- current selected frame - int
# <xyz_trajectory>.n_frames		-- number of total frames in trajectory - int
# <xyz_trajectory>.atoms_per_frame	-- number of atoms per timestep - list(N)
# <xyz_trajectory>.content		-- raw content of file as list(str)
#---------------------------------------------------

class xyz_trajectory:
    
    def __init__(self, filepath):
        self.path = filepath
        self.content = self.__file_content()
        self.n_lines = len(self.content)
        self.atoms_per_frame, self.comments = self.__get_traj_properties()
        self.start_line_per_frame = numpy.hstack(([0], numpy.cumsum(numpy.array(self.atoms_per_frame) + 2)[:-1]))
        self.n_frames = len(self.atoms_per_frame)
        self.__iter__()

    def __file_content(self):
        with open(self.path) as f:
            contents = f.readlines()
        return contents
    
    def __get_traj_properties(self):
        atoms_per_frame = list()
        comments = list()
        line = 0
        while line < self.n_lines:
            atoms_per_frame.append(int(self.content[line]))
            comments.append(self.content[line+1].split())
            line = numpy.cumsum(numpy.array(atoms_per_frame)+2)[-1]
        return atoms_per_frame, comments

    def positions(self):
        frame = self.content[self.start_line_per_frame[self.frame]+2:self.start_line_per_frame[self.frame]+self.atoms_per_frame[self.frame]+2]
        return numpy.array([line.split() for line in frame])[:,1:4].astype(float)

    def atom_types(self):    
        frame = self.content[self.start_line_per_frame[self.frame]+2:self.start_line_per_frame[self.frame]+self.atoms_per_frame[self.frame]+2]
        return numpy.array([line.split() for line in frame])[:,0].astype(str)
    
    def __iter__(self):
        self.frame = 0
        return self

    def __next__(self):
        if self.frame < self.n_frames-1:
            a = self.frame
            self.frame += 1
            return self
        else:
            raise StopIteration
    def __getitem__(self, x):
        self.frame = x
        return self

#---------------------------------------------------
# Class for trajectories in .xyz-fileformat.
# Supports variable number of atoms per frame.
#
# Class is iterable and subscriptable by the n-frames
#
# functions:
#
# <lammpstrj>.data() 		-- data of the selected frame - numpy.array(M,N); N = number of atoms; M = number of logged properties
# <lammpstrj>.columns()  	-- columnnames of the logged properties - list(M)
# <lammpstrj>.cell()		-- pbcs of the selected frame  - list([xlo, xhi], [ylo, yhi], [zlo, zhi])
# <lammpstrj>.df_frame()	-- pandas DataFrame containg data+columns of selected timestep df(M,N)
#
# porperties:
#
# <lammpstrj>.frame		-- current selected frame - int
# <lammpstrj>.n_frames		-- number of total frames in trajectory - int
# <lammpstrj>.atoms_per_frame	-- number of atoms per timestep - list()
# <lammpstrj>.timesteps	-- timestep of the logged frames - list()
# <lammpstrj>.content		-- raw content of file as list()
#---------------------------------------------------

class lammpstrj:

    def __init__(self, filepath):
        self.path = filepath
        self.content = self.__file_content()
        self.n_lines = len(self.content)
        self.timesteps, self.atoms_per_frame = self.__get_traj_properties()
        self.start_line_per_frame = numpy.hstack(([0], numpy.cumsum(numpy.array(self.atoms_per_frame) + 9)[:-1]))
        self.n_frames = len(self.atoms_per_frame)
        self.__iter__()
        self.__print()

    def data(self):
        frame = self.content[self.start_line_per_frame[self.frame]+9:self.start_line_per_frame[self.frame]+self.atoms_per_frame[self.frame]+9]
        return numpy.array([line.split() for line in frame]).astype(float)        

    def cell(self):
        frame = self.content[self.start_line_per_frame[self.frame]+5:self.start_line_per_frame[self.frame]+8]
        return numpy.array([i.split() for i in frame]).astype(float).tolist()

    def columns(self):
        frame = self.content[self.start_line_per_frame[self.frame]+8]        
        return frame.split()[2:]  
    
    def df_frame(self):
        return pandas.DataFrame(data=self.data(), columns=self.columns())

    def __file_content(self):
        with open(self.path) as f:
            contents = f.readlines()
        return contents

    def __get_traj_properties(self):
        timesteps = list()
        atoms_per_frame = list()
        line = 0
        while line < self.n_lines:
            timesteps.append(int(self.content[line+1]))
            atoms_per_frame.append(int(self.content[line+3]))
            line = numpy.cumsum(numpy.array(atoms_per_frame)+9)[-1]
        return timesteps, atoms_per_frame 

    def __print(self):
        print('Loaded LAMMPS trajectory file {file} containing {frames} frames.'.format(file=self.path, frames=self.n_frames))

    def __iter__(self):
        self.frame = 0
        return self

    def __next__(self):
        if self.frame < self.n_frames-1:
            a = self.frame
            self.frame += 1
            return self
        else:
            raise StopIteration
    def __getitem__(self, x):
        self.frame = x
        return self


# Filters the pandas dataframe which contains atom data by deleting the 
# types given in the id list
# Input:  d » atom data | Pandas DataFrame 
#         ids » atom types to delete | List
# Output: filtered atom data | Pandas DataFrame

def sel_data(atoms_data, ids):
    sel_data = pandas.DataFrame()
    sel_data = sel_data.append(atoms_data)
    for i in ids:
        sel_data = sel_data[(sel_data['type']!=i)]
    
    return sel_data

# Load any compressed pickle file
def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data
    
def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        pickle.dump(data, f)

# loads and returns a pickled objects
def loosen(file):
    pikd = open(file, 'rb')
    data = pickle.load(pikd, encoding='latin1')
    pikd.close()
    return data

# Saves the "data" with the "title" and adds the .pickle
def full_pickle(title, data):
    pikd = open(title + '.pickle', 'wb')
    pickle.dump(data, pikd)
    pikd.close()  
   
#Reads data from LAMMPS of fix ave/time file
def read_ave_time(filename):

    print('Reading file {}'.format(filename))
    df_data = pandas.read_csv(filename)
    df_data.set_axis(['i'], axis = 1, inplace = True)
    df_data = df_data.i.str.split(expand=True)
    header = list(df_data.iloc[0,1:].values)
    header.append('del')
    df_data.set_axis(header, axis = 1, inplace = True)
    df_data = df_data.iloc[1:]
    df_data.reset_index(drop=True, inplace=True)

    df_data = df_data.drop(['del'], axis=1)
    df_data = df_data.apply(pandas.to_numeric)
    print('File processed: {}'.format(filename))

    return df_data

#Reads data from lammps chunk density files
# returns two dataframes: 
# first: whole data
# second: averaged data by all frames
def chunk_dens(filename):
    
    df_data = pandas.read_csv(filename)
    df_data.set_axis(['chunk'], axis = 1, inplace = True)
    df_data = df_data.chunk.str.split(expand=True)
    df_data = df_data.iloc[2:]
    df_data = df_data.drop([4], axis=1)
    
    df_data = df_data.apply(pandas.to_numeric)
    df_data.reset_index(drop=True, inplace=True)
    df_data.set_axis(["chunk", "coord", "Ncount", "dens"], axis = 1, inplace = True)
    
    n_chunks = df_data.iloc[0,1]
    lines = n_chunks+1
    chunk_list = list(range(1, int(lines), 1))
    df_avg = pandas.DataFrame()
    coords = df_data.coord[1:int(lines)].values
    
    df_avg['chunk'] = chunk_list
    df_avg['coord'] = coords
    df_avg['dens'] = numpy.nan
    df_avg['std'] = numpy.nan
    
    for i in chunk_list:
        df_avg.loc[i-1, 'dens'] = numpy.mean(df_data.dens[df_data.chunk==i])
        df_avg.loc[i-1, 'std'] = numpy.std(df_data.dens[df_data.chunk==i])

    return df_data, df_avg
    
# Reads data from LAMMPS fix ave/chunk file
# Variable number of chunks supported

def read_ave_chunk(filename):
    print('Reading file: {}'.format(filename))
    df_data = pandas.read_csv(filename)
    df_data.set_axis(['i'], axis = 1, inplace = True)
    df_data = df_data.i.str.split(expand=True)
    header = numpy.array(df_data.iloc[2,0:3].values).astype(float)
    column_names = list(df_data.iloc[1,1:].values)
    df_data = df_data.iloc[2:,:-1]
    df_data.reset_index(drop = True, inplace = True)
    df_data.columns = column_names
    df_data = df_data.apply(pandas.to_numeric)
    number_of_chunks = list()
    timesteps = list()
    number_of_chunks.append(int(df_data.iloc[0,1]))
    timesteps.append(int(df_data.iloc[0,0]))
    current_line = 0
    while current_line < len(df_data)-(number_of_chunks[-1]+1):
        current_line = current_line+(number_of_chunks[-1]+1)
        number_of_chunks.append(int(df_data.iloc[current_line,1]))
        timesteps.append(int(df_data.iloc[current_line,0]))
    lines_per_timestep = numpy.array(number_of_chunks)+1
    end_indicies = numpy.cumsum(lines_per_timestep)
    start_indicies = scipy.ndimage.interpolation.shift(end_indicies, 1, cval = 0)
    data_dict = {}
    i = 0
    while i < len(start_indicies):
         data_dict[timesteps[i]]=df_data.iloc[start_indicies[i]+1:end_indicies[i]]
         i = i + 1
    print('File processed: {}'.format(filename))

    return data_dict

#---------------------- Benni Smooth ------------------------------------------  

def smooth(x, lookahead=21, window='flat', pol_order=2):
    """ Glaettung der Funktion

    Parameter
    __________
    y           ... Eingabevektor
    lookahead   ... Bereich [lookahead/2 ... Zahl ... l/2] aus dem der Mittelwerte gebildet wird
    window      ... Auswahl der Window-Function (flat, Savgol, hamming, bartlett, blackman, hanning, kaiser,...)
    pol_order   ... Polynomgrad der Aproximationsfunktion für den SavGol filter
    """

    # Korrektur der Eingaben
    if lookahead % 2 == 0:
        lookahead += 1

    aa = int(lookahead // 2)

    # Faltung des Vektors fuer Mittelwerte am Vektorrand
    s = numpy.r_[x[lookahead - 1:0:-1], x, x[-2:-lookahead - 1:-1]]

    # Auswahl des Mittelwertalgorythmus
    if window == 'SavGol':
        y = savgol_filter(x, lookahead, pol_order)
    else:
        if window == 'flat':  # moving average
            w = numpy.ones(lookahead, 'd')
        else:
            w = eval('ws.' + window + '(lookahead)')

        # Bildung Mittelwerte
        y = numpy.convolve(w / w.sum(), s, mode='valid')

        # Beschneiden des Vektors
        y = y[aa:-aa]

    return y
 
def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out
 
 
#---------------------- PLOTS ------------------------------------------------  
# Plots the atom unit density of atom cords in a pandas dataframe
# except the atoms with the atom types in the given list ids = []
# xlo, and xhi are the box and plot dimensions for pbc
def contour_cell(atoms_data, ids, xlo, xhi):
    sel_data = pandas.DataFrame()
    sel_data = sel_data.append(atoms_data)
    for i in ids:
        sel_data = sel_data[(sel_data['type']!=i)]
        
    sel_data_pbc1 = pandas.DataFrame()
    sel_data_pbc2 = pandas.DataFrame()
    sel_data_pbc1 = sel_data_pbc1.append(sel_data[sel_data['x']<xlo+(xhi-xlo)/2])
    sel_data_pbc2 = sel_data_pbc2.append(sel_data[sel_data['x']>xlo+(xhi-xlo)/2])
    sel_data_pbc1.x = sel_data_pbc1.x+(xhi-xlo)
    sel_data_pbc2.x = sel_data_pbc2.x-(xhi-xlo)
    
    sel_data_ext = pandas.DataFrame()
    sel_data_ext = sel_data_ext.append(sel_data)
    sel_data_ext = sel_data_ext.append(sel_data_pbc1)
    sel_data_ext = sel_data_ext.append(sel_data_pbc2)
    
    x = numpy.array(sel_data_ext.x)
    y = numpy.array(sel_data_ext.y)
    z = numpy.array(sel_data_ext.z)
    
    k = gaussian_kde(numpy.vstack([x, z]))
    k.set_bandwidth(bw_method=k.factor / 3.)
    xi, zi = numpy.mgrid[x.min():x.max():x.size**0.5*1j,z.min():z.max():z.size**0.5*1j]
    yi = k(numpy.vstack([xi.flatten(), zi.flatten()]))
    
    fig,ax=plt.subplots(1,1)
    ax.set_xlim(xlo, xhi)
    
    cp = ax.contourf(xi, zi, yi.reshape(xi.shape))
    fig.colorbar(cp)
    ax.set_title('Filled Contours Plot')
    ax.set_ylabel('y (Angs)')
    plt.show()


def split_lines_of_timeframe(timeframe_array):
    new_array = numpy.array([0]*len(timeframe_array[0].split()))
    for i in timeframe_array:
        line = numpy.array(i.split()).astype(float)
        new_array = numpy.vstack((new_array, line))
    new_array = numpy.delete(new_array, 0, axis=0)

    return new_array

#---------------------------------------------------
# Function to read lammps trajectories - NOT RECOMMEND
#
# !!!! ATTENTION: VERY MEMORY DEMANDING !!!!
#    !!!! USE lammpstrj class instead !!!!
#
# multiprocessing enabled, using 4 cores by default
# returns dictionary
# for data see read_lammps_dump(filename).keys()
#---------------------------------------------------

def read_lammps_dump(filename, cpu = 4):
    print('reading dump file: {}'.format(filename))
    with open(filename) as f:
        lines = f.read().splitlines()
    f.close()

    n_cores = cpu
    number_header_lines = 9
    line_current_timestep = 1
    line_atom_count_in_header = 3
    line_box_boundaries = 4
    line_box_x_header = 5
    line_box_y_header = 6
    line_box_z_header = 7
    first_header = lines[0:number_header_lines]

    atoms_per_timestep = list()
    timesteps_list = list()
    box_size_list = list()

    box_boundaries = lines[line_box_boundaries].split()[3:6]
    atoms_per_timestep.append(int(lines[line_atom_count_in_header]))
    timesteps_list.append(int(lines[line_current_timestep]))
    box_x = numpy.array(lines[line_box_x_header].split()).astype(float)
    box_y = numpy.array(lines[line_box_y_header].split()).astype(float)
    box_z = numpy.array(lines[line_box_z_header].split()).astype(float)
    box_size = numpy.array([box_x, box_y, box_z])
    box_size_list.append(box_size)

    atom_parameters = first_header[8].split()[2:11]

    current_line = 0
    while current_line < len(lines)-(atoms_per_timestep[-1]+number_header_lines):
        atoms_line = current_line+atoms_per_timestep[-1]+number_header_lines+line_atom_count_in_header
        timestep_line = current_line+atoms_per_timestep[-1]+number_header_lines+line_current_timestep
        atoms_per_timestep.append(int(lines[atoms_line]))
        timesteps_list.append(int(lines[timestep_line]))
        box_x_line = current_line+atoms_per_timestep[-1]+number_header_lines+line_box_x_header
        box_y_line = current_line+atoms_per_timestep[-1]+number_header_lines+line_box_y_header
        box_z_line = current_line+atoms_per_timestep[-1]+number_header_lines+line_box_z_header
        box_x = numpy.array(lines[box_x_line].split()).astype(float)
        box_y = numpy.array(lines[box_y_line].split()).astype(float)
        box_z = numpy.array(lines[box_z_line].split()).astype(float)
        box_size = numpy.array([box_x, box_y, box_z])
        box_size_list.append(box_size)
        current_line = current_line + atoms_per_timestep[-1] + number_header_lines

    print('found {} timesteps'.format(len(timesteps_list)))
    print('atom parameters {}'.format(atom_parameters))

    atoms_per_timestep = numpy.array(atoms_per_timestep)
    lines_per_timestep = atoms_per_timestep+number_header_lines

    end_line_per_timestep = list()
    index_headers = list()
    i = 0
    while i < len(lines_per_timestep):
        if len(end_line_per_timestep)==0:
            pre_lines = 0
        if len(end_line_per_timestep)>0:
            pre_lines = end_line_per_timestep[-1]
        end_line_per_timestep.append(lines_per_timestep[i]+pre_lines)
        i = i+1
    start_line_per_timestep = shift(end_line_per_timestep, 1, cval=0)

    for i in start_line_per_timestep:
        index_headers = index_headers+list(numpy.arange(i,i+number_header_lines,1))

    index_atom_data = list(set(range(len(lines)))-set(index_headers))
    lines = numpy.array(lines)

    atom_lines = lines[index_atom_data]
    header_lines = lines[index_headers]

    atom_lines_split = numpy.array_split(atom_lines, numpy.cumsum(atoms_per_timestep))
    del atom_lines_split[-1]

    print('start processing timesteps')
    print('using {} cpu cores'.format(n_cores))
    pool = mp.Pool(n_cores)
    results = pool.map(split_lines_of_timeframe, [atom_lines_split[i] for i in range(0, len(atom_lines_split), 1)])
    pool.close()

    return_dict = {}
    return_dict['timestep_data'] = results
    return_dict['timesteps'] = timesteps_list
    return_dict['number_of_atoms'] = atoms_per_timestep
    return_dict['atom_paramters'] = atom_parameters
    return_dict['box_sizes'] = box_size_list
    return_dict['box_boundaries'] = box_boundaries

    print('processing complete!')
    return return_dict
    
    

def read_xyz(file, header_lines=2):
    df_data = pandas.read_csv(file, names=(['i']), header=None)
    df_data = df_data.i.str.split(expand=True)
    n_atoms_per_step = list()
    line = 0
    while line < len(df_data):
        n_atoms_per_step.append(int(df_data.iloc[line, 0]))
        line = line + n_atoms_per_step[-1] + header_lines

    lines_per_timestep = numpy.array(n_atoms_per_step) + header_lines
    start_line_per_timestep = numpy.cumsum(lines_per_timestep)
    start_line_per_timestep = numpy.hstack(([0], start_line_per_timestep[:-1]))

    ts_data = {}
    for i in range(len(start_line_per_timestep)):
        ts_i = int(df_data.loc[start_line_per_timestep[i]+1].values[2])
        ts_coords = pandas.DataFrame().append(df_data[start_line_per_timestep[i] + header_lines:start_line_per_timestep[i]+lines_per_timestep[i]])
        ts_coords.columns=(['atom', 'x', 'y', 'z'])
        ts_coords.loc[:,(['x','y','z'])] = ts_coords.loc[:,(['x','y','z'])].apply(pandas.to_numeric)
        ts_coords.reset_index(inplace=True, drop=True)
        ts_data[ts_i] = ts_coords
    
    return ts_data

#---------------------------------------------------
# Function to read lammps.log files.
# Returns a dictionary with n_runs, according to the
# number of "run" commands in the lammps-input script.
# To avoid problems with multiple timesteps by 
# using reset_timestep for example.
#---------------------------------------------------

def read_log(log_filename):
    log_file = open(log_filename, 'r')
    lines = log_file.readlines()
    count = 0
    add = 0

    run_log_dict = {}
    run = 0

    for line in lines:
        txt_line = line.strip()
        if 'Step' in txt_line:
            run_log_dict[run] = list()
            add = 1
            parameters_i = txt_line.split()
            run_log_dict[run].append(txt_line.split())
        if 'Loop' in txt_line:
            add = 0
            run = run + 1
        if add == 1:
            if txt_line.split()[0].isnumeric():
                run_log_dict[run].append(txt_line.split())

    df_log_dict = {}
    for i in list(run_log_dict.keys()):
        log_i = run_log_dict[i]
        names_i = run_log_dict[i][0]
        data_i = run_log_dict[i][1:]
        df = pandas.DataFrame(data_i, columns=names_i)
        df = df.apply(pandas.to_numeric, errors='coerce')
        df_log_dict[i] = df
        
    return df_log_dict
