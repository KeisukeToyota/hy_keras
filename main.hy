(import [__future__ [print_function]])
(import keras)
(import [keras.datasets [mnist]])
(import [keras.models [Sequential]])
(import [keras.layers [Dense
                       Dropout
                       Flatten
                       Conv2D
                       MaxPooling2D]])
(import [keras [backend :as K]])

(setv batch_size 128)
(setv num_classes 10)
(setv epochs 12)

(setv img_rows 28
      img_cols 28)

(setv [(, x_train y_train) (, x_test y_test)] (mnist.load_data))

(if (= (K.image_data_format) "channels_first")
        (setv x_train (x_train.reshape (first x_train.shape) 1 img_rows img_cols)
              x_test (x_test.reshape (first x_test.shape) 1 img_rows img_cols)
              input_shape (, 1 img_rows img_cols))
        (setv x_train (x_train.reshape (first x_train.shape) img_rows img_cols 1)
              x_test (x_test.reshape (first x_test.shape) img_rows img_cols 1)
              input_shape (, img_rows img_cols 1)))

(setv x_train (x_train.astype "float32")
      x_test (x_test.astype "float32"))

(/= x_train 255)
(/= x_test 255)

(print "x_train shape:" x_train.shape)
(print (first x_train.shape) "train samples")
(print (first x_test.shape) "test samples")

(setv y_train (keras.utils.to_categorical y_train num_classes)
      y_test (keras.utils.to_categorical y_test num_classes))

(setv model (Sequential))
(model.add (Conv2D 32 :kernel_size (, 3 3)
                   :activation "relu"
                   :input_shape input_shape))
(model.add (Conv2D 64 (, 3 3) :activation "relu"))
(model.add (MaxPooling2D :pool_size (, 2 2)))
(model.add (Dropout 0.25))
(model.add (Flatten))
(model.add (Dense 128 :activation "relu"))
(model.add (Dropout 0.5))
(model.add (Dense num_classes :activation "softmax"))

(model.compile :loss keras.losses.categorical_crossentropy
               :optimizer (keras.optimizers.Adadelta)
               :metrics ["accuracy"])

(model.fit x_train y_train
           :batch_size batch_size
           :epochs epochs
           :verbose 1
           :validation_data (, x_test y_test))

(setv score (model.evaluate x_test y_test :verbose 0))

(print "Test loss:" (first score))
(print "Test accuracy:" (second score))
