import torch
import torch.optim as optim
import torch.nn.functional as F

class Weight_Tuning():
    def __init__(self,learning_rate):
        self.learning_rate = learning_rate

   
    def update_weights_LG_UA(self, model, inputs, targets, category_threshold, ua_threshold):
        optimizer = optim.SGD(model.parameters(), lr=self.learning_rate)
        prev_loss = None
    
        for epoch in range(category_threshold):
            optimizer.zero_grad()
            predictions = model(inputs)
            loss = F.cross_entropy(predictions, targets)
            loss.backward()
            optimizer.step()
        
            # Check if learning rate is less than ua_threshold
            if self.learning_rate <= ua_threshold:
                return
        
            # Check if previous loss is greater than current loss
            if prev_loss is not None and loss >= prev_loss:
                self.learning_rate = self.learning_rate * 1.2
                for i, w in enumerate(model.parameters()):
                    w.data = weight_backup[i]
                model.train()
                return self.update_weights_LG_UA(model, inputs, targets, ua_threshold)
        
            prev_loss = loss
            weight_backup = [w.data.clone() for w in model.parameters()]
            self.learning_rate = self.learning_rate * 0.7
            model.train()

    def update_weights_EB_LG_UA(self, model, inputs, targets, num_epochs, batch_size, category_threshold, ua_threshold):
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        prev_loss = float('inf')
        for epoch in range(num_epochs):
            for i in range(0, len(inputs), batch_size):
                input_batch = inputs[i:i+batch_size]
                target_batch = targets[i:i+batch_size]
                optimizer.zero_grad()
                predictions = model(input_batch)
                loss = torch.nn.functional.cross_entropy(predictions, target_batch)
                loss.backward()
                gradients = [p.grad for p in model.parameters()]
                if loss >= prev_loss:
                    self.learning_rate *= 1.2
                    model.load_state_dict(prev_state)
                    optimizer.param_groups[0]['lr'] = self.learning_rate
                    break
                else:
                    prev_state = model.state_dict()
                    prev_loss = loss
                    optimizer.step()
                    if epoch > category_threshold and loss >= prev_loss:
                        return
                    if self.learning_rate < ua_threshold:
                        model.load_state_dict(prev_state)
                        optimizer.param_groups[0]['lr'] *= 0.7
                        break
