import tensorflow as tf

class WeightTuning():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        
    def update_weights_LG(self, model, inputs, targets, category_threshold):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        for epoch in range(category_threshold):
            with tf.GradientTape() as tape:
                predictions = model(inputs)
                loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(targets, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    def update_weights_EB(self, model, inputs, targets, num_epochs, batch_size):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        for epoch in range(num_epochs):
            for i in range(0, len(inputs), batch_size):
                input_batch = inputs[i:i+batch_size]
                target_batch = targets[i:i+batch_size]
                with tf.GradientTape() as tape:
                    predictions = model(input_batch)
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(target_batch, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    
    def update_weights_EB_LG(self, model, inputs, targets, num_epochs, batch_size, category_threshold):
        optimizer = tf.keras.optimizers.SGD(learning_rate=self.learning_rate)
        for epoch in range(num_epochs):
            for i in range(0, len(inputs), batch_size):
                input_batch = inputs[i:i+batch_size]
                target_batch = targets[i:i+batch_size]
                with tf.GradientTape() as tape:
                    predictions = model(input_batch)
                    loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)(target_batch, predictions)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                if epoch > category_threshold and loss >= prev_loss:
                    return
                prev_loss = loss
