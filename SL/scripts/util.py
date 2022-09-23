import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

np.random.seed(6)

VALIDATION_SPLIT = 0.25

def get_data(data_set):
	if data_set == "gender":
		df = pd.read_csv("../data/gender.csv", header=0)
		df['gender'] = LabelEncoder().fit_transform(df['gender'])
		data = df.drop('gender', axis=1).values
		target = df.get('gender').values
		target = target.reshape((len(target), 1))
		x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=VALIDATION_SPLIT, shuffle=True)
		x_test_small = []
		y_test_small = []
		n_v, n_f = 0, 0
		for class_id, features in zip(y_test, x_test):
			if class_id == 1:
				n_f += 1
				x_test_small.append(features)
				y_test_small.append(class_id)
			elif n_v < n_f:
				n_v += 1
				x_test_small.append(features)
				y_test_small.append(class_id)
		x_test = x_test_small
		y_test = y_test_small
	elif data_set == "creditcard":
		df = pd.read_csv("../data/creditcard.csv", header=0)
		data = df.drop('Class', axis=1).values
		target = df.get('Class').values
		target = target.reshape((len(target), 1))
		x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=VALIDATION_SPLIT, shuffle=True)
		x_test_small = []
		y_test_small = []
		n_v, n_f = 0, 0
		for class_id, features in zip(y_test, x_test):
			if class_id == 1:
				n_f += 1
				x_test_small.append(features)
				y_test_small.append(class_id)
			elif n_v < n_f:
				n_v += 1
				x_test_small.append(features)
				y_test_small.append(class_id)
		x_test = x_test_small
		y_test = y_test_small
	else:
		raise ValueError("%s data set is not implemented" % data_set)
	print("Selected data set is", data_set, "with", len(y_train) + len(y_test), "data (train:", len(y_train), ", test:", len(y_test), ")")
	x_train, x_test = normalize_x_data(np.array(x_train), np.array(x_test))
	return x_train, np.array(y_train), x_test, np.array(y_test)

def normalize_x_data(x_train, x_test):
	# compute normalization on training set only
	x_min = np.min(x_train, axis=0)
	x_max = np.max(x_train, axis=0)
	# apply the same transform on both datasets
	epsilon = 0.0001
	x_train[:] = x_train[:] - x_min / (x_max - x_min + epsilon)
	x_test[:] = x_test[:] - x_min / (x_max - x_min + epsilon)
	return x_train, x_test

def consolidate_dict_data(dict_data, consolidate_argx, consolidate_argy, consolidate_argz, argx_name="arg_x", argy_name="arg_y", filters=None):
	consolidate_dict = dict()
	for keys, values in dict_explore(dict_data):
		arg_x = keys[consolidate_argx]
		arg_y = keys[consolidate_argy]
		arg_z = keys[consolidate_argz]
		arg_xy = (arg_x, arg_y)
		if filters is None or are_keys_on_filters(keys, filters):
			if arg_z not in consolidate_dict:
				consolidate_dict[arg_z] = dict()
			if arg_xy not in consolidate_dict[arg_z]:
				consolidate_dict[arg_z][arg_xy] = list()
			consolidate_dict[arg_z][arg_xy].append(values)
	return_dict = dict()
	for arg_z, d in consolidate_dict.items():
		data = []
		for k, l in d.items():
			a = np.array(l)
			mean = np.mean(a)
			std = np.std(a)
			median = np.median(a)
			a_min = np.min(a)
			a_max = np.max(a)
			data.append([k[0], k[1], mean, median, std, a_min, a_max])
		df = pd.DataFrame(data, columns=[argx_name, argy_name, "mean", "median", "std", "min", "max"])
		return_dict[arg_z] = df
	return return_dict

def are_keys_on_filters(keys, filters):
	for key_index, key_value in filters:
		try:
			if keys[key_index] not in key_value:
				return False
		except TypeError:
			if keys[key_index] != key_value:
				return False
	return True

def dict_explore(dict_data, keys=None):
	if keys is None:
		keys = []
	if type(dict_data) == dict:
		for k, v in dict_data.items():
			keys.append(k)
			yield from dict_explore(v, keys)
			keys.pop()
	else:
		yield keys, dict_data