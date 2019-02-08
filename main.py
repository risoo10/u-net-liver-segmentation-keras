from processing import *
from model import *
from keras.callbacks import ModelCheckpoint

# Process data
# process_data()

# Load data
inputs, labels = load_data()

# Normalize and reshape
inputs = normalize_data(inputs).reshape(inputs.shape[0], COLUMS, ROWS, 1)
labels = labels.reshape(labels.shape[0], COLUMS, ROWS, 1)

print('[inputs] min:', np.min(inputs), 'max:', np.max(inputs))
print('[labels] min:', np.min(labels), 'max:', np.max(labels))

# Split data for train / validation
indices = np.random.permutation(len(inputs))
split = int(0.7 * len(indices))
print('Train / Validation split (', split, ',', len(indices) - split, ')')

train_inputs = inputs[indices[:split]]
test_inputs = inputs[indices[split:]]

train_labels = labels[indices[:split]]
test_labels = labels[indices[split:]]

# Model
model = u_net((COLUMS, ROWS, 1))
model_checkpoint = ModelCheckpoint('unet_model.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit(train_inputs, train_labels, batch_size=2, epochs=20)






