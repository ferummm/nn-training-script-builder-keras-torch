import os
from {model_name} import Model
import sys
import json
from lib.mypckg import dataloader 

epochs = {epochs}
batch_size = {batch_size}
dst_directory = '{dst_directory}'
weight_path = os.path.join(os.path.normpath(dst_directory),'{model_name}.h5')
history_path = os.path.join(os.path.normpath(dst_directory),'{model_name}_history.json')
fn_data = '{filename_data}'

#Load dataset
x_data, y_data = dataloader.load_data(filename=fn_data, 
                                      res_cols={res_cols}, 
                                      drop_na = {drop_na}, 
                                      col_names = {col_names}, 
                                      inp_shape = {input_shape}, 
                                      out_shape = {output_shape},
                                      reshape_single_output = True)


train_size = int(len(x_data)*{split_size}*0.01)
val_size = int((len(x_data)-train_size)/2)

x_train, y_train = x_data[:train_size], y_data[:train_size]
x_test, y_test = x_data[train_size:], y_data[train_size:]
x_val, y_val = x_test[:val_size], y_test[:val_size]

#Create model
model = Model()

#Compile the model
model.compile(optimizer='adam', loss='{loss_func}', metrics=['{metric}'])

#Train the model
history = model.fit(x_train, y_train, 
                    epochs=epochs, 
                    batch_size=batch_size, 
                    verbose = 2,
                    validation_data=(x_val, y_val))

history_dict = dict()
history_dict['loss'] = history.history['loss']
history_dict['val_loss'] = history.history['val_loss']
history_dict['{metric}'] = history.history['{metric}'] 
history_dict['val_{metric}'] = history.history['val_{metric}']

# Save history to json file
with open(history_path, 'w') as f:
    json.dump(history_dict, f)

# Evaluate the model on the test data
print('\n Evaluate on test data')
results = model.evaluate(x_test[val_size:], y_test[val_size:], batch_size=batch_size, verbose = 2)
print("test loss, test {metric}:", results)

# Generate predictions the output of the last layer
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3])
print("predictions shape:", predictions.shape)

print('Keras model saved in: ', weight_path) 
model.save_weights(weight_path,save_format="h5")
