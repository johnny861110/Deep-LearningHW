import tensorflow as tf

class WeightTuning():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    
    
    def update_weights_LG(self, model, inputs, targets, category_threshold):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        for epoch in range(category_threshold):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
    
    def update_weights_LG_UA(self, model, inputs, targets,category_threshold,ua_threshold):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        prev_loss = float('inf')
        for epoch in range(category_threshold):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(targets, predictions)  
            
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Check if learning rate is less than ua_threshold
            if self.learning_rate <= ua_threshold:
                return
            
            # Check if previous loss is greater than current loss
            if prev_loss is not None and loss >= prev_loss:
                self.learning_rate = self.learning_rate * 1.2
                for i, w in enumerate(model.trainable_variables):
                    w.assign(weight_backup[i])
                model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=optimizer)
                return self.update_weights_LG_UA(model, inputs, targets,ua_threshold)
            
            prev_loss = loss
            weight_backup = [w.numpy() for w in model.trainable_variables]
            self.learning_rate = self.learning_rate * 0.7
            model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), optimizer=optimizer)
            
    
    
    def update_weights_EB(self, model, inputs, targets, num_epochs, batch_size):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        for epoch in range(num_epochs):
            for i in range(0, len(inputs), batch_size):
                input_batch = inputs[i:i+batch_size]
                target_batch = targets[i:i+batch_size]
                with tf.GradientTape() as tape:
                    predictions = model(input_batch)
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(target_batch, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    
    
    def update_weights_EB_LG(self, model, inputs, targets, num_epochs, batch_size, category_threshold):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        history = {'loss': [], 'accuracy': []}  # initialize history
        prev_loss = float('inf')
        for epoch in range(num_epochs):
            for i in range(0, len(inputs), batch_size):
                input_batch = inputs[i:i+batch_size]
                target_batch = targets[i:i+batch_size]
                with tf.GradientTape() as tape:
                    predictions = model(input_batch)
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(target_batch, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                history['loss'].append(loss.numpy())
                history['accuracy'].append(tf.keras.metrics.categorical_accuracy(target_batch, predictions).numpy().mean())
                if epoch > category_threshold and loss >= prev_loss:
                    return history
                prev_loss = loss
         # evaluate model on test data
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
        print(f'Test accuracy: {test_acc}')
        return history
    
    
            
    def update_weights_EB_LG_UA(self, model, inputs, targets, num_epochs, batch_size, category_threshold,ua_threshold):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        prev_loss = float('inf')
        for epoch in range(num_epochs):
            for i in range(0, len(inputs), batch_size):
                input_batch = inputs[i:i+batch_size]
                target_batch = targets[i:i+batch_size]
                with tf.GradientTape() as tape:
                    predictions = model(input_batch)
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(target_batch, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                if loss >= prev_loss:
                    self.learning_rate *= 1.2
                    model.set_weights(prev_weights)
                    optimizer.learning_rate = self.learning_rate
                    break
                else:
                    prev_weights = model.get_weights()
                    prev_loss = loss
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                    if epoch > category_threshold and loss >= prev_loss:
                        return
                    if self.learning_rate < ua_threshold:
                        model.set_weights(prev_weights)
                        optimizer.learning_rate *= 0.7
                        break

    