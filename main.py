from processing import *
from model import *
from keras.callbacks import ModelCheckpoint, TensorBoard

# Process data
# process_data()

# Model traing and evaluation
filename = 'unet_model-light-51ep'
model_path = 'models/' + filename + '.h5'

if os.path.exists(model_path):
    # Load evaluation data
    print('Loading saved model [', filename, '] ....')
    eval_inputs = pickle.load(open('test-inputs.np', 'rb')).reshape((-1, COLUMS, ROWS, 1))[:200]
    eval_labels = pickle.load(open('test-labels.np', 'rb')).reshape((-1, COLUMS, ROWS, 1))[:200]

    # Load saved model
    model = load_model(model_path)
    model.summary()

    # Evaluate on test data
    results = model.evaluate(eval_inputs, eval_labels, batch_size=2)
    print('Evaluation results')
    for i, metric in enumerate(model.metrics_names):
        print(metric, ':', results[i])

    # Plot predictions
    ind = 80
    pred = model.predict(np.stack([eval_inputs[ind]]))
    plt.subplot(131)
    plt.imshow(eval_inputs[ind].reshape((COLUMS, ROWS)), cmap=plt.cm.bone)
    plt.subplot(132)
    plt.imshow(eval_labels[ind].reshape((COLUMS, ROWS)))
    plt.subplot(133)
    plt.imshow(pred.reshape((COLUMS, ROWS)))
    plt.show()

else:
    # Load data
    inputs, labels = load_data()

    # Normalize and reshape
    inputs = normalize_data(inputs).reshape(inputs.shape[0], COLUMS, ROWS, 1)
    labels = labels.reshape(labels.shape[0], COLUMS, ROWS, 1)

    print('[inputs] min:', np.min(inputs), 'max:', np.max(inputs))
    print('[labels] min:', np.min(labels), 'max:', np.max(labels))

    # Split data for train / validation
    indices = np.random.permutation(len(inputs))
    split = int(0.85 * len(indices))
    print('Train / Validation split (', split, ',', len(indices) - split, ')')

    train_inputs = inputs[indices[:split]]
    test_inputs = inputs[indices[split:]]

    train_labels = labels[indices[:split]]
    test_labels = labels[indices[split:]]

    # Train and save model
    model = u_net((COLUMS, ROWS, 1))
    model.summary()
    model_checkpoint = ModelCheckpoint(filename + 'hdf5', monitor='loss', verbose=1, save_best_only=True)
    model.fit(train_inputs, train_labels, batch_size=8, epochs=51, validation_data=(test_inputs, test_labels),
              callbacks=[TensorBoard(log_dir="logs/" + filename, histogram_freq=0, write_graph=True)])
    model.save(model_path)




