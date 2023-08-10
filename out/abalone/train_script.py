import os
from abaloneModel import Model
import sys
import json
from lib.mypckg import dataloader 

epochs = 10
batch_size = 25
dst_directory = 'C:/Users/Admin/Desktop/VS/project/main/out/abalone'
weight_path = os.path.join(os.path.normpath(dst_directory),'abaloneModel.h5')
history_path = os.path.join(os.path.normpath(dst_directory),'abaloneModel_history.json')
fn_data = 'C:/Users/Admin/Desktop/VS/project/main/data/abalone_data.xlsx'

#Load dataset
x_data, y_data = dataloader.load_data(filename=fn_data, 
                                      res_cols=[7], 
                                      drop_na = False, 
                                      col_names = True, 
                                      inp_shape = (1, 7), 
                                      out_shape = (1, 1),
                                      reshape_single_output = True)


train_size = int(len(x_data)*80.0*0.01)
val_size = int((len(x_data)-train_size)/2)

x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]
x_val, y_val = x_test[:val_size], y_test[:val_size]

#Create model
model = Model()

#Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

#Train the model
history = model.fit(x_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose = 2,
                    validation_data=(x_val, y_val))

history_dict = dict()
history_dict['loss'] = history.history['loss']
history_dict['val_loss'] = history.history['val_loss']
history_dict['mae'] = history.history['mae'] 
history_dict['val_mae'] = history.history['val_mae']

# Save history to json file
with open(history_path, 'w') as f:
    json.dump(history_dict, f)

# Evaluate the model on the test data
print('\n Evaluate on test data')
results = model.evaluate(x_test[val_size:], y_test[val_size:], batch_size=batch_size, verbose = 2)
print("test loss, test mae:", results)

# Generate predictions the output of the last layer
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)

print('Keras model saved in: ', weight_path) 
model.save_weights(weight_path,save_format="h5")
