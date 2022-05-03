# load data and consider a single feature (Economy..GDP.per.capita) and the output to be estimated (happiness)
import csv
import os
import matplotlib.pyplot as plt
import numpy as np


def load_data(filename, input_var_name1, input_var_name2, output_var_name):
    data = []
    data_names = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                data_names = row
            else:
                data.append(row)
            line_count += 1
    selected_var1 = data_names.index(input_var_name1)
    selected_var2 = data_names.index(input_var_name2)

    inputs = [[float(data[i][selected_var1]), float(data[i][selected_var2])] for i in range(len(data))]
    selected_output = data_names.index(output_var_name)
    outputs = [float(data[i][selected_output]) for i in range(len(data))]

    return inputs, outputs


crt_dir = os.getcwd()
file_path = os.path.join(crt_dir, 'data', 'v1-world-happiness-report-2017.csv')

inputs, outputs = load_data(file_path, 'Economy..GDP.per.Capita.', "Freedom", 'Happiness.Score')
print('in:  ', inputs[:5])
print('out: ', outputs[:5])

# Split the Data Into Training and Test Subsets
indexes = [i for i in range(len(inputs))]
train_sample = np.random.choice(indexes, int(0.8 * len(inputs)), replace = False)
validation_sample = [i for i in indexes if not i in train_sample]

trainInputs = [inputs[i] for i in train_sample]
trainOutputs = [outputs[i] for i in train_sample]

validationInputs = [inputs[i] for i in validation_sample]
validationOutputs = [outputs[i] for i in validation_sample]


from myRegression import MyLinearBivariateRegression

regressor = MyLinearBivariateRegression()

w0, w1 = regressor.fit(trainInputs, trainOutputs)

print('the learnt model: f(x) = ', w0, ' + ', w1[0], ' * x1' + ' + ', w1[1], ' * x2')

# makes predictions for test data
computedValidationOutputs = regressor.predict(validationInputs)

# compute the differences between the predictions and real outputs
error = 0.0
for t1, t2 in zip(computedValidationOutputs, validationOutputs):
    error += (t1 - t2) ** 2
error = error / len(validationOutputs)
print('loss : ', error)